// Dumps Tesseract's raw LSTM softmax outputs and its beam-search result WITH the model's LSTM
// dictionary, the way LSTMRecognizer::RecognizeLine runs it in normal recognition
// (tesseract-gpu dictionary parity). Also decodes the same outputs without a dictionary.
//
// usage: tess_dump_dict <lang.traineddata> <lang> <out_dir> <line.png>...
//
// The model is loaded with LSTMRecognizer::Load(params, lang, mgr), which calls LoadDictionary:
// Dict::SetupForLoad + LoadLSTM (lstm-punc-dawg, lstm-word-dawg, lstm-number-dawg) + FinishLoad,
// exactly as Tesseract::init_tesseract_lang_data does with lstm_use_matrix=1. The dictionary
// search is RecodeBeamSearch(recoder, null_char, SimpleTextOutput(), rec.GetDict()), decoded
// with kDictRatio 2.25, kCertOffset -0.085 and worst_dict_cert = kWorstDictCertainty /
// kCertaintyScale = -25.0f / 7.0f, the constants of lstmrecognizer.cpp and linerec.cpp.
//
// writes <out_dir>/<basename>.dict.bin (all int32 little-endian unless noted):
//   T, C, float32 probs[T*C]
//   n0, ids_nodict[n0]                    RecodeBeamSearch with dict == nullptr
//   n, ids_dict[n], xcoords[n]            ExtractBestPathAsUnicharIds with the dictionary
//   nw, then per word: len, permuter, nbytes, utf8 text[nbytes]
//                                         ExtractBestPathAsWords with the dictionary
// and prints "<png>\t<dict text>" per line (words joined by single spaces).
#include <allheaders.h>
#include <cstdio>
#include <string>
#include <vector>
#include "dict.h"
#include "imagedata.h"
#include "lstmrecognizer.h"
#include "networkio.h"
#include "pageres.h"
#include "params.h"
#include "recodebeam.h"
#include "tessdatamanager.h"
using namespace tesseract;

static void put_i(FILE *f, int v) { fwrite(&v, 4, 1, f); }
static void put_iv(FILE *f, const std::vector<int> &v) {
  put_i(f, v.size());
  if (!v.empty()) fwrite(v.data(), 4, v.size(), f);
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: tess_dump_dict model lang out_dir png...\n"); return 2; }
  TessdataManager mgr;
  if (!mgr.Init(argv[1])) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  ParamsVectors params;
  LSTMRecognizer rec;
  if (!rec.Load(&params, argv[2], &mgr)) { fprintf(stderr, "load failed\n"); return 1; }
  if (rec.GetDict() == nullptr) { fprintf(stderr, "no lstm dictionary loaded\n"); return 1; }
  RecodeBeamSearch dsearch(rec.GetRecoder(), rec.null_char(), rec.SimpleTextOutput(),
                           rec.GetDict());
  RecodeBeamSearch nsearch(rec.GetRecoder(), rec.null_char(), rec.SimpleTextOutput(), nullptr);
  const double kDictRatio = 2.25, kCertOffset = -0.085;
  const float kCertaintyScale = 7.0f, kWorstDictCertainty = -25.0f;
  const double worst_dict_cert = kWorstDictCertainty / kCertaintyScale;
  for (int i = 4; i < argc; ++i) {
    Pix *p = pixRead(argv[i]);
    if (!p) { fprintf(stderr, "cannot read %s\n", argv[i]); continue; }
    ImageData im(false, Image(p));
    float scale = 0;
    NetworkIO inputs, outputs;
    if (!rec.RecognizeLine(im, 0.0f, false, false, false, &scale, &inputs, &outputs)) {
      printf("%s\t<FAIL>\n", argv[i]);
      continue;
    }
    std::vector<int> ids0, xs0, ids, xs;
    std::vector<float> certs, ratings;
    nsearch.Decode(outputs, kDictRatio, kCertOffset, worst_dict_cert, &rec.GetUnicharset(), 0);
    nsearch.ExtractBestPathAsUnicharIds(false, &rec.GetUnicharset(), &ids0, &certs, &ratings, &xs0);
    dsearch.Decode(outputs, kDictRatio, kCertOffset, worst_dict_cert, &rec.GetUnicharset(), 0);
    dsearch.ExtractBestPathAsUnicharIds(false, &rec.GetUnicharset(), &ids, &certs, &ratings, &xs);
    xs.resize(ids.size());  // drop the trailing width sentinel
    tesseract::PointerVector<tesseract::WERD_RES> words;
    tesseract::TBOX box(0, 0, pixGetWidth(p), pixGetHeight(p));
    dsearch.ExtractBestPathAsWords(box, scale, false, &rec.GetUnicharset(), &words, 0);

    std::string base = argv[i];
    base = base.substr(base.find_last_of('/') + 1);
    std::string out = std::string(argv[3]) + "/" + base + ".dict.bin";
    FILE *f = fopen(out.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
    int T = outputs.Width(), C = outputs.NumFeatures();
    put_i(f, T);
    put_i(f, C);
    for (int t = 0; t < T; ++t) fwrite(outputs.f(t), 4, C, f);
    put_iv(f, ids0);
    put_iv(f, ids);
    if (!xs.empty()) fwrite(xs.data(), 4, xs.size(), f);
    put_i(f, words.size());
    std::string text;
    for (int w = 0; w < words.size(); ++w) {
      const tesseract::WERD_CHOICE *bc = words[w]->best_choice;
      std::string s = bc->unichar_string();
      put_i(f, bc->length());
      put_i(f, bc->permuter());
      put_i(f, s.size());
      fwrite(s.data(), 1, s.size(), f);
      if (!text.empty()) text += " ";
      text += s;
    }
    fclose(f);
    printf("%s\t%s\n", argv[i], text.c_str());
  }
  return 0;
}
