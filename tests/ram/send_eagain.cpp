// Exercises FFmpegRamEncoder's send/receive boundary with scripted FFmpeg
// return values, covering the EAGAIN flow-control contract: EAGAIN on
// either side of the encoder must never surface as an encode error, while
// real failures must still propagate. RAM entry (ffmpeg_ram_encode) drives
// the production encode() chain with a real registered frame.

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <cassert>
#include <cstdio>
#include <deque>
#include <initializer_list>
#include <memory>
#include <vector>

#ifdef NDEBUG
#error These tests require assertions.
#endif

namespace {
const AVFrame *expected_frame;
int64_t expected_ms = 0;
int send_error = 0;
int send_calls = 0;
int callbacks = 0;
std::deque<int> receive_results;

int injected_send(AVCodecContext *, const AVFrame *frame) {
  ++send_calls;
  assert(frame == expected_frame);
  assert(frame->pts == expected_ms);
  return send_error;
}

int injected_receive(AVCodecContext *, AVPacket *packet) {
  assert(!receive_results.empty());
  int result = receive_results.front();
  receive_results.pop_front();
  if (result < 0)
    return result;
  if (result == 0) {
    assert(av_new_packet(packet, 1) == 0);
    packet->data[0] = 42;
    packet->pts = expected_ms;
    return 0;
  }
  // A successful receive with no packet data exercises the empty-packet
  // guard.
  av_packet_unref(packet);
  return 0;
}
} // namespace

// Compile the production methods with only the FFmpeg I/O boundary replaced.
#define avcodec_send_frame injected_send
#define avcodec_receive_packet injected_receive
#include "../../cpp/ffmpeg_ram/ffmpeg_ram_encode.cpp"
#undef avcodec_send_frame
#undef avcodec_receive_packet

extern "C" void hwcodec_log(int, const char *) {}
extern "C" void hwcodec_av_log_callback(int, const char *) {}

namespace {
void output(const uint8_t *data, int size, int64_t, int, const void *) {
  assert(size == 1 && data[0] == 42);
  ++callbacks;
}

void prepare(int result, std::initializer_list<int> prepare_receive) {
  send_error = result;
  send_calls = 0;
  callbacks = 0;
  expected_ms += 100;
  receive_results.assign(prepare_receive);
}
} // namespace

int main() {
  using EncoderPtr =
      std::unique_ptr<FFmpegRamEncoder, decltype(&ffmpeg_ram_free_encoder)>;
  EncoderPtr encoder(
      new FFmpegRamEncoder("h264_videotoolbox", "", 64, 32,
                           AV_PIX_FMT_YUV420P, 0, 30, 60, 0, 0, 0, 0, 1, -1,
                           output),
      ffmpeg_ram_free_encoder);
  encoder->c_ = avcodec_alloc_context3(nullptr);
  assert(encoder->c_);
  encoder->frame_ = av_frame_alloc();
  assert(encoder->frame_);
  encoder->frame_->format = AV_PIX_FMT_YUV420P;
  encoder->frame_->width = 64;
  encoder->frame_->height = 32;
  assert(av_frame_get_buffer(encoder->frame_, 0) == 0);
  encoder->pkt_ = av_packet_alloc();
  assert(encoder->pkt_);
  expected_frame = encoder->frame_;

  // fill_frame() reads offsets from the encoder the same way init() would.
  const int *ls = encoder->frame_->linesize;
  encoder->offset_[0] = ls[0] * 32;
  encoder->offset_[1] = encoder->offset_[0] + ls[1] * 32 / 2;
  std::vector<uint8_t> yuv(encoder->offset_[1] + ls[2] * 32 / 2, 0);

  auto encode_once = [&]() {
    return ffmpeg_ram_encode(encoder.get(), yuv.data(), (int)yuv.size(),
                             nullptr, expected_ms);
  };

  // Baseline: one packet comes out, then the drain hits EAGAIN.
  prepare(0, {0, AVERROR(EAGAIN)});
  assert(encode_once() == 0);
  assert(send_calls == 1);
  assert(callbacks == 1);

  // Warm-up: the frame was accepted but no packet is ready yet.
  prepare(0, {AVERROR(EAGAIN)});
  assert(encode_once() == 0);
  assert(callbacks == 0);

  // Input queue full: send returns EAGAIN, a packet is still pending and
  // must be delivered by the drain loop instead of failing the encode.
  prepare(AVERROR(EAGAIN), {0, AVERROR(EAGAIN)});
  assert(encode_once() == 0);
  assert(send_calls == 1);
  assert(callbacks == 1);

  // Input queue full while still warming up: both sides EAGAIN, still no
  // error.
  prepare(AVERROR(EAGAIN), {AVERROR(EAGAIN)});
  assert(encode_once() == 0);
  assert(callbacks == 0);

  // Input queue full with several queued packets: all of them drain.
  prepare(AVERROR(EAGAIN), {0, 0, AVERROR(EAGAIN)});
  assert(encode_once() == 0);
  assert(callbacks == 2);

  // A real send failure propagates.
  prepare(AVERROR(EIO), {});
  assert(encode_once() == AVERROR(EIO));
  assert(callbacks == 0);

  // A real receive failure propagates.
  prepare(0, {AVERROR(EIO)});
  assert(encode_once() == -1);
  assert(callbacks == 0);

  // Empty packet on receive keeps being an error (guard).
  prepare(0, {1});
  assert(encode_once() == -1);
  assert(callbacks == 0);

  std::printf("ram send/receive EAGAIN flow control: all checks passed\n");
  return 0;
}
