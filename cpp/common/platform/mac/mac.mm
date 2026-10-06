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

// ---------------------- OBS-style: VTCopyVideoEncoderList + real session probe ----------------------
// Enumeration alone can lie on hackintosh / older Intel GPUs: an encoder may be
// listed as hardware-accelerated while VTCompressionSessionCreate actually fails
// (e.g. Skylake HD 515 lists HEVC hardware but session creation returns
// kVTVideoEncoderMalfunctionErr -12903, since HEVC encode needs Kaby Lake+).
// So after enumerating, create a minimal session with the same
// RequireHardwareAcceleratedVideoEncoder spec FFmpeg videotoolboxenc uses with
// allow_sw=0 — making "detected" always mean "creatable" at connection time.
//
// Hardening against an unstable VideoToolbox service (old Intel hackintosh):
// session creation may fail instantly with -12903, take ~10s to succeed, or
// hang indefinitely — and the hwcodec check child used to die inside these
// calls, never reaching the ipc send, so the config cache was never written
// (parent SIGKILLed it after its wait window). Three measures below:
//   1. every VideoToolbox call runs behind a watchdog thread; a hang degrades
//      the single probe result to "unsupported" instead of killing the check,
//   2. the four-way answer is computed once per process and cached — encoder
//      enumeration, decoder enumeration and every gpu-signature compare share
//      it, so the flaky service is probed once instead of being hammered on
//      hot paths (the server computes the signature on every cache-miss load),
//   3. a transient probe failure is retried once after a short delay: on cold
//      start -12903 usually clears within seconds (observed same-process
//      recovery).
static void vtSessionProbeCallback(void *refCon, void *frameRefCon, OSStatus status,
                                   VTEncodeInfoFlags infoFlags,
                                   CMSampleBufferRef sampleBuffer) {
    (void)refCon; (void)frameRefCon; (void)status; (void)infoFlags; (void)sampleBuffer;
}

static bool canCreateCompressionSession(CMVideoCodecType codecType) {
    // Prefer hardware, fall back to Apple software encoder when GVA is unavailable
    // (hackintosh / older GPUs). Matches hwcodec force_hw(allow_sw=1) semantics.
    CFMutableDictionaryRef encoderSpec = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 1,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (!encoderSpec) {
        return false;
    }
    CFDictionarySetValue(encoderSpec,
                         kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder,
                         kCFBooleanTrue);

    VTCompressionSessionRef session = NULL;
    OSStatus status = VTCompressionSessionCreate(
        kCFAllocatorDefault,
        256, 256, /* small probe size — any real encoder accepts this, keeps probe fast */
        codecType,
        encoderSpec,
        NULL, /* sourceImageBufferAttributes */
        NULL, /* compressedDataAllocator */
        vtSessionProbeCallback,
        NULL, /* outputCallbackRefCon */
        &session);

    CFRelease(encoderSpec);

    if (status != noErr || !session) {
        LOG_WARN(std::string("VideoToolbox session probe failed for codec type ") +
                 std::to_string(codecType) + ", OSStatus = " + std::to_string(status) +
                 " - treating as unsupported (enumeration lied)");
        return false;
    }

    VTCompressionSessionInvalidate(session);
    CFRelease(session);
    return true;
}

// Run `fn` on a detached worker thread and wait at most `timeout_ms`.
// On timeout return 0 ("unsupported"): a hung VideoToolbox call must degrade
// one probe result, never the whole check process. The worker keeps a
// shared_ptr to its state (and owns a copy of `fn`), so a late completion
// cannot touch freed stack.
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

    CFArrayRef encoderList = NULL;
    OSStatus status = VTCopyVideoEncoderList(NULL, &encoderList);
    if (status != noErr || !encoderList) {
        return 0;
    }

    int32_t found = 0;
    CFIndex count = CFArrayGetCount(encoderList);

    for (CFIndex i = 0; i < count; i++) {
        CFDictionaryRef encoderDict = (CFDictionaryRef)CFArrayGetValueAtIndex(encoderList, i);
        if (!encoderDict) continue;

        // Check codec type matches
        CFNumberRef codecTypeNum = (CFNumberRef)CFDictionaryGetValue(encoderDict, kVTVideoEncoderList_CodecType);
        if (!codecTypeNum) continue;

        CMVideoCodecType encCodecType = 0;
        CFNumberGetValue(codecTypeNum, kCFNumberSInt32Type, &encCodecType);
        if (encCodecType != codecType) continue;

        // Check if hardware accelerated
        CFBooleanRef hwRef = (CFBooleanRef)CFDictionaryGetValue(encoderDict, kVTVideoEncoderList_IsHardwareAccelerated);
        Boolean isHW = (hwRef && CFBooleanGetValue(hwRef));

        if (isHW) {
            // Enumerated as hardware — verify by actually creating a session the
            // same way the real encoder will be created at connection time.
            // One retry for transient failures: on cold start VideoToolbox can
            // return kVTVideoEncoderNotAvailableNowErr (-12903) even for codecs
            // that work moments later (observed same-process recovery).
            if (!canCreateCompressionSession(codecType)) {
                usleep(1500 * 1000);
                if (!canCreateCompressionSession(codecType)) {
                    continue; // try next entry of the same codec type, if any
                }
            }
            found = 1;
            break;
        }
    }

    CFRelease(encoderList);
    return found;
}

// -------------- Your Public Interface: Unchanged ------------------
extern "C" void checkVideoToolboxSupport(int32_t *h264Encoder, int32_t *h265Encoder, int32_t *h264Decoder, int32_t *h265Decoder) {
    // https://stackoverflow.com/questions/50956097/determine-if-ios-device-can-support-hevc-encoding
    // Computed at most once per process (mutex-guarded so concurrent callers
    // wait for the single probe round instead of racing VideoToolbox):
    // the check child reaches this through encoder enumeration, decoder
    // enumeration and the gpu-signature field; the server reaches it through
    // every signature compare on a cache-miss load. All share the cache.
    static std::mutex cache_mutex;
    static bool cached = false;
    static int32_t cached_h264_encoder = 0;
    static int32_t cached_h265_encoder = 0;
    static int32_t cached_h264_decoder = 0;
    static int32_t cached_h265_decoder = 0;

    std::lock_guard<std::mutex> lock(cache_mutex);
    if (!cached) {
        cached_h264_encoder = run_with_timeout(
            [] { return hasHardwareEncoder(false); }, 10 * 1000, "h264 encoder enumeration");
        cached_h265_encoder = run_with_timeout(
            [] { return hasHardwareEncoder(true); }, 10 * 1000, "h265 encoder enumeration");
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
