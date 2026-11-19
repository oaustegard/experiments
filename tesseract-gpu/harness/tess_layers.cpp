// Dumps the output of every top-level layer of a Tesseract LSTM network for one line image.
// usage: tess_layers <lang.traineddata> <out_prefix> <line.png>
// writes <out_prefix>.L<i>.bin: int32 H, W, F, int_mode, float32 data[H*W*F] (y-major, x, f)
#include <allheaders.h>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>
#include <cmath>
#define private public
#define protected public
#include "imagedata.h"
#include "lstmrecognizer.h"
#include "networkio.h"
#include "networkscratch.h"
#include "params.h"
#include "series.h"
#include "tessdatamanager.h"
using namespace tesseract;

static void dump(const std::string &path, const NetworkIO &io) {
  FILE *f = fopen(path.c_str(), "wb");
  const StrideMap &sm = io.stride_map();
  int H = sm.Size(FD_HEIGHT), W = sm.Size(FD_WIDTH), F = io.NumFeatures(), im = io.int_mode() ? 1 : 0;
  fwrite(&H, 4, 1, f); fwrite(&W, 4, 1, f); fwrite(&F, 4, 1, f); fwrite(&im, 4, 1, f);
  for (int t = 0; t < io.Width(); ++t)
    for (int k = 0; k < F; ++k) {
      float v = im ? io.i(t)[k] : io.f(t)[k];
      fwrite(&v, 4, 1, f);
    }
  fclose(f);
}

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  TessdataManager mgr;
  if (!mgr.Init(argv[1])) return 1;
  ParamsVectors params;
  LSTMRecognizer rec;
  if (!rec.Load(&params, "", &mgr)) return 1;
  Pix *p = pixRead(argv[3]);
  ImageData im(false, Image(p));
  float scale = 0;
  NetworkIO inputs, outputs;
  rec.RecognizeLine(im, 0.0f, false, false, false, &scale, &inputs, &outputs);
  auto *series = static_cast<Series *>(rec.network_);
  rec.SetRandomSeed();
  NetworkScratch scratch;
  NetworkIO cur = inputs;
  dump(std::string(argv[2]) + ".in.bin", cur);
  for (size_t i = 0; i < series->stack_.size(); ++i) {
    NetworkIO out;
    series->stack_[i]->Forward(false, cur, nullptr, &scratch, &out);
    dump(std::string(argv[2]) + ".L" + std::to_string(i) + ".bin", out);
    cur = out;
  }
  dump(std::string(argv[2]) + ".final.bin", outputs);
  {
    rec.SetRandomSeed();
    auto *cs = static_cast<Series *>(series->stack_[1]);
    NetworkIO conv;
    cs->stack_[0]->Forward(false, inputs, nullptr, &scratch, &conv);
    dump(std::string(argv[2]) + ".conv.bin", conv);
    printf("sample_iteration %d seed %lld\n", rec.sample_iteration_, (long long)rec.randomizer_.seed_);
  }
  return 0;
}
