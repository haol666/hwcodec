#include <AVFoundation/AVFoundation.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreMedia/CoreMedia.h>
#include <MacTypes.h>
#include <VideoToolbox/VideoToolbox.h>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <ratio>
#include <string>
#include <sys/_types/_int32_t.h>
#include <sys/event.h>
#include <thread>
#include <unistd.h>
#include "../../log.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// ---------------------- Core: More Robust Hardware Encoder Detection ----------------------

// Run `fn` on a detached worker thread and wait at most `timeout_ms`. On timeout
// return 0 ("unsupported"): on some older Intel machines a VideoToolbox probe call
// can block indefinitely, and one hung probe must degrade only its own result,
// never the whole hwcodec check child (which delivers its result over ipc and is
// SIGKILLed by a parent watchdog if it hangs). The worker owns a copy of `fn` and
// keeps a shared_ptr to its state, so a late completion cannot touch freed stack.
static int run_with_timeout(const std::function<int()> fn, int timeout_ms, const char *what) {
    struct Shared {
        std::atomic<int> done;
        int value;
        Shared() : done(0), value(0) {}
    };
    std::shared_ptr<Shared> shared = std::make_shared<Shared>();
    std::thread worker([shared, fn]() {
        shared->value = fn();
        shared->done.store(1, std::memory_order_release);
    });
    worker.detach();
    for (int waited = 0; waited < timeout_ms; waited += 50) {
        if (shared->done.load(std::memory_order_acquire)) {
            return shared->value;
        }
        usleep(50 * 1000);
    }
    if (shared->done.load(std::memory_order_acquire)) {
        return shared->value;
    }
    LOG_WARN(std::string("VideoToolbox probe timed out after ") +
             std::to_string(timeout_ms) + "ms: " + what +
             " - treating as unsupported");
    return 0;
}
static int32_t hasHardwareEncoder(bool h265) {
    CMVideoCodecType codecType = h265 ? kCMVideoCodecType_HEVC : kCMVideoCodecType_H264;

    // ---------- Path A: Quick Query with Enable + Require ----------
    // Note: Require implies Enable, but setting both here makes it easier to bypass the strategy on some models that default to a software encoder.
    CFMutableDictionaryRef spec = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                                            &kCFTypeDictionaryKeyCallBacks,
                                                            &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(spec, kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder, kCFBooleanTrue);
    CFDictionarySetValue(spec, kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder, kCFBooleanTrue);

    CFDictionaryRef properties = NULL;
    CFStringRef outID = NULL;

    // Use 1280x720 for capability detection to reduce the probability of "no hardware encoding" due to resolution/level issues.
    OSStatus result = VTCopySupportedPropertyDictionaryForEncoder(1280, 720, codecType, spec, &outID, &properties);

    if (properties) CFRelease(properties);
    if (outID) CFRelease(outID);
    if (spec) CFRelease(spec);

    if (result == noErr) {
        // Explicitly found an encoder that meets the "hardware-only" specification.
        return 1;
    }
    // Reaching here means either no encoder satisfying Require was found (common), or another error occurred.
    // For all failure cases, continue with the safer "session-level confirmation" path to avoid misjudgment.

    // ---------- Path B: Create Session and Read UsingHardwareAcceleratedVideoEncoder ----------
    CFMutableDictionaryRef enableOnly = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                                                  &kCFTypeDictionaryKeyCallBacks,
                                                                  &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(enableOnly, kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder, kCFBooleanTrue);

    VTCompressionSessionRef session = NULL;
    // Also use 1280x720 to reduce profile/level interference
    OSStatus st = VTCompressionSessionCreate(kCFAllocatorDefault,
                                             1280, 720, codecType,
                                             enableOnly,      /* encoderSpecification */
                                             NULL,            /* sourceImageBufferAttributes */
                                             NULL,            /* compressedDataAllocator */
                                             NULL,            /* outputCallback */
                                             NULL,            /* outputRefCon */
                                             &session);
    if (st != noErr || !session) {
        // A cold VideoToolbox service can fail a creatable encoder with a
        // transient kVTVideoEncoderNotAvailableNowErr (-12903) that clears
        // within seconds; retry once before concluding there is no hardware.
        usleep(1500 * 1000);
        st = VTCompressionSessionCreate(kCFAllocatorDefault,
                                        1280, 720, codecType,
                                        enableOnly,      /* encoderSpecification */
                                        NULL,            /* sourceImageBufferAttributes */
                                        NULL,            /* compressedDataAllocator */
                                        NULL,            /* outputCallback */
                                        NULL,            /* outputRefCon */
                                        &session);
    }
    if (enableOnly) CFRelease(enableOnly);

    if (st != noErr || !session) {
        // Creation failed, considered no hardware available.
        return 0;
    }

    // First, explicitly prepare the encoding process to give VideoToolbox a chance to choose between software/hardware.
    OSStatus prepareStatus = VTCompressionSessionPrepareToEncodeFrames(session);
    if (prepareStatus != noErr) {
        VTCompressionSessionInvalidate(session);
        CFRelease(session);
        return 0;
    }

    // Query the session's read-only property: whether it is using a hardware encoder.
    CFBooleanRef usingHW = NULL;
    st = VTSessionCopyProperty(session,
                               kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder,
                               kCFAllocatorDefault,
                               (void **)&usingHW);

    Boolean isHW = (st == noErr && usingHW && CFBooleanGetValue(usingHW));

    if (usingHW) CFRelease(usingHW);
    VTCompressionSessionInvalidate(session);
    CFRelease(session);

    return isHW ? 1 : 0;
}

// -------------- Your Public Interface: Unchanged ------------------
extern "C" void checkVideoToolboxSupport(int32_t *h264Encoder, int32_t *h265Encoder, int32_t *h264Decoder, int32_t *h265Decoder) {
    // https://stackoverflow.com/questions/50956097/determine-if-ios-device-can-support-hevc-encoding
    // Computed at most once per process (mutex-guarded): the check child reaches
    // this through encoder enumeration, decoder enumeration and the gpu-signature
    // field, and the RustDesk server reaches it through every signature compare on
    // a cache-miss config load. One probe round serves all callers instead of
    // hammering an unstable VideoToolbox service; a hung probe degrades only its
    // own result (see run_with_timeout above).
    static std::mutex cache_mutex;
    static bool cached = false;
    static int32_t cached_h264_encoder = 0;
    static int32_t cached_h265_encoder = 0;
    static int32_t cached_h264_decoder = 0;
    static int32_t cached_h265_decoder = 0;

    std::lock_guard<std::mutex> lock(cache_mutex);
    if (!cached) {
        cached_h264_encoder = run_with_timeout(
            [] { return hasHardwareEncoder(false); }, 10 * 1000, "h264 encoder detection");
        cached_h265_encoder = run_with_timeout(
            [] { return hasHardwareEncoder(true); }, 10 * 1000, "h265 encoder detection");
        cached_h264_decoder = run_with_timeout(
            [] { return VTIsHardwareDecodeSupported(kCMVideoCodecType_H264) ? 1 : 0; },
            5 * 1000, "h264 decode support query");
        cached_h265_decoder = run_with_timeout(
            [] { return VTIsHardwareDecodeSupported(kCMVideoCodecType_HEVC) ? 1 : 0; },
            5 * 1000, "h265 decode support query");
        cached = true;
    }

    *h264Encoder = cached_h264_encoder;
    *h265Encoder = cached_h265_encoder;
    *h264Decoder = cached_h264_decoder;
    *h265Decoder = cached_h265_decoder;

    return;
}

extern "C" uint64_t GetHwcodecGpuSignature() {
    int32_t h264Encoder = 0;
    int32_t h265Encoder = 0;
    int32_t h264Decoder = 0;
    int32_t h265Decoder = 0;
    checkVideoToolboxSupport(&h264Encoder, &h265Encoder, &h264Decoder, &h265Decoder);
    return (uint64_t)h264Encoder << 24 | (uint64_t)h265Encoder << 16 | (uint64_t)h264Decoder << 8 | (uint64_t)h265Decoder;
}

static void *parent_death_monitor_thread(void *arg) {
  int kq = (intptr_t)arg;
  struct kevent events[1];

  int ret = kevent(kq, NULL, 0, events, 1, NULL);

  if (ret > 0) {
    // Parent process died, terminate this process
    LOG_INFO("Parent process died, terminating hwcodec check process");
    exit(1);
  }

  return NULL;
}

extern "C" int setup_parent_death_signal() {
  // On macOS, use kqueue to monitor parent process death
  pid_t parent_pid = getppid();
  int kq = kqueue();

  if (kq == -1) {
    LOG_DEBUG("Failed to create kqueue for parent monitoring");
    return -1;
  }

  struct kevent event;
  EV_SET(&event, parent_pid, EVFILT_PROC, EV_ADD | EV_ONESHOT, NOTE_EXIT, 0,
         NULL);

  int ret = kevent(kq, &event, 1, NULL, 0, NULL);

  if (ret == -1) {
    LOG_ERROR("Failed to register parent death monitoring on macOS\n");
    close(kq);
    return -1;
  } else {

    // Spawn a thread to monitor parent death
    pthread_t monitor_thread;
    ret = pthread_create(&monitor_thread, NULL, parent_death_monitor_thread,
                         (void *)(intptr_t)kq);

    if (ret != 0) {
      LOG_ERROR("Failed to create parent death monitor thread");
      close(kq);
      return -1;
    }

    // Detach the thread so it can run independently
    pthread_detach(monitor_thread);
    return 0;
  }
}
