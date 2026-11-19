// Dumps Tesseract's raw LSTM softmax outputs and its no-dictionary beam-search text for line
// images, using the library's own LSTMRecognizer and RecodeBeamSearch (tesseract-gpu parity).
// usage: tess_dump <lang.traineddata> <out_dir> <line.png>...
// writes <out_dir>/<basename>.bin: int32 T, int32 C, float32 probs[T*C], int32 n, int32 unichar_ids[n]
// and prints "<png>\t<text>" per line (words joined by single spaces, as tesseract prints them).
#include <allheaders.h>
#include <cstdio>
#include <string>
#include <vector>
#include "imagedata.h"
#include "lstmrecognizer.h"
#include "networkio.h"
#include "params.h"
#include "pageres.h"
#include "recodebeam.h"
#include "tessdatamanager.h"
using namespace tesseract;

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage\n"); return 2; }
  TessdataManager mgr;
  if (!mgr.Init(argv[1])) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  ParamsVectors params;
  LSTMRecognizer rec;
  if (!rec.Load(&params, "", &mgr)) { fprintf(stderr, "load failed\n"); return 1; }
  RecodeBeamSearch search(rec.GetRecoder(), rec.null_char(), rec.SimpleTextOutput(), nullptr);
  const double kDictRatio = 2.25, kCertOffset = -0.085;
  const float worst_dict_cert = -25.0f / 7.0f;
  for (int i = 3; i < argc; ++i) {
    Pix *p = pixRead(argv[i]);
    if (!p) { fprintf(stderr, "cannot read %s\n", argv[i]); continue; }
    ImageData im(false, Image(p));
    float scale = 0;
    NetworkIO inputs, outputs;
    if (!rec.RecognizeLine(im, 0.0f, false, false, false, &scale, &inputs, &outputs)) {
      printf("%s\t<FAIL>\n", argv[i]);
      continue;
    }
    search.Decode(outputs, kDictRatio, kCertOffset, worst_dict_cert, &rec.GetUnicharset(), 0);
    std::vector<int> ids, xs;
    std::vector<float> certs, ratings;
    search.ExtractBestPathAsUnicharIds(false, &rec.GetUnicharset(), &ids, &certs, &ratings, &xs);
    tesseract::PointerVector<tesseract::WERD_RES> words;
    tesseract::TBOX box(0, 0, pixGetWidth(p), pixGetHeight(p));
    search.ExtractBestPathAsWords(box, scale, false, &rec.GetUnicharset(), &words, 0);
    std::string text;
    for (int w = 0; w < words.size(); ++w) {
      if (!text.empty()) text += " ";
      text += words[w]->best_choice->unichar_string();
    }
    std::string base = argv[i];
    base = base.substr(base.find_last_of('/') + 1);
    std::string out = std::string(argv[2]) + "/" + base + ".bin";
    FILE *f = fopen(out.c_str(), "wb");
    int T = outputs.Width(), C = outputs.NumFeatures();
    fwrite(&T, 4, 1, f);
    fwrite(&C, 4, 1, f);
    for (int t = 0; t < T; ++t) fwrite(outputs.f(t), 4, C, f);
    int n = ids.size();
    fwrite(&n, 4, 1, f);
    fwrite(ids.data(), 4, n, f);
    // network input: Width() timesteps (height*width) x features, int8 or float, as float32
    int TI = inputs.Width(), FI = inputs.NumFeatures(), imode = inputs.int_mode() ? 1 : 0;
    fwrite(&TI, 4, 1, f);
    fwrite(&FI, 4, 1, f);
    fwrite(&imode, 4, 1, f);
    for (int t = 0; t < TI; ++t) {
      for (int k = 0; k < FI; ++k) {
        float v = imode ? inputs.i(t)[k] : inputs.f(t)[k];
        fwrite(&v, 4, 1, f);
      }
    }
    fclose(f);
    printf("%s\t%s\n", argv[i], text.c_str());
  }
  return 0;
}
