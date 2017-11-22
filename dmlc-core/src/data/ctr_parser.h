/*!
 *  Copyright (c) 2015 by Contributors
 * \file ctr_parser.h
 * \brief iterator parser to parse ctr format
 * \author wangzhulei
 */
#ifndef DMLC_DATA_CTR_PARSER_H_
#define DMLC_DATA_CTR_PARSER_H_

#include <dmlc/data.h>
#include <dmlc/parameter.h>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include "./row_block.h"
#include "./text_parser.h"
#include "./strtonum.h"

namespace dmlc {
namespace data {

struct CTRParserParam : public Parameter<CTRParserParam> {
  std::string format;
  int label_column;
  // declare parameters
  DMLC_DECLARE_PARAMETER(CTRParserParam) {
    DMLC_DECLARE_FIELD(format).set_default("ctr")
        .describe("File format.");
    DMLC_DECLARE_FIELD(label_column).set_default(-1)
        .describe("Column index that will put into label.");
  }
};


/*!
 * \brief CTRParser, parses a dense ctr format.
 *  Currently is a dummy implementation, when label column is not specified.
 *  All columns are treated as real dense data.
 *  label will be assigned to 0.
 *
 *  This should be extended in future to accept arguments of column types.
 */
template <typename IndexType>
class CTRParser : public TextParserBase<IndexType> {
 public:
  typedef  std::unordered_map<uint32_t, uint32_t> t_uint32_uint32_map;
  typedef  std::unordered_map<uint32_t, std::vector<float> > t_uint32_vector_map;
  t_uint32_uint32_map feature_index_map;
  t_uint32_vector_map feature_normalize_map;
  int load_feature_index(const char * feature_index){
     const uint32_t MAX_NORMAL_LINE_LENGTH = 8192;
     char buf[MAX_NORMAL_LINE_LENGTH];
     char * field_list[3];
     uint32_t field_num;
     FILE* fp = fopen(feature_index,"rt");
     if (fp == NULL){
         LOG(INFO) << "mxnet Cannot find feature_index";
         return -1;
     }
  
     while (fgets(buf,MAX_NORMAL_LINE_LENGTH,fp)){
         split_string('\t',buf,3,field_list,&field_num);
         if (field_num != 2){
            LOG(INFO) << "feature_index can't meet";
            return -1;
         }
         uint32_t feature_set_id = atoi(field_list[0]);
         uint32_t feature_index = atoi(field_list[1]);
         feature_index_map[feature_index] = feature_set_id;
     }
  
     return 0;
  }
  int load_feature_normalize(const char * feature_normalize){
     const uint32_t MAX_NORMAL_LINE_LENGTH = 8192;
     char buf[MAX_NORMAL_LINE_LENGTH];
     char * field_list[8];
     uint32_t field_num;
     FILE* fp = fopen(feature_normalize,"rt");
     if (fp == NULL){
         LOG(INFO) << "mxnet Cannot find feature_normalize";
         return -1;
     }
  
     while (fgets(buf,MAX_NORMAL_LINE_LENGTH,fp)){
         split_string('\t',buf,8,field_list,&field_num);
         if (field_num != 7){
            LOG(INFO) << "feature_normalize can't meet";
            return -1;
         }
         uint32_t feature_set_id = atoi(field_list[0]);
         float max_ctr = atof(field_list[1]);
         float min_ctr = atof(field_list[2]);
         float max_ec = atof(field_list[3]);
         float min_ec = atof(field_list[4]);
         float max_coec = atof(field_list[5]);
         float min_coec = atof(field_list[6]);
         feature_normalize_map[feature_set_id].push_back(max_ctr);
         feature_normalize_map[feature_set_id].push_back(min_ctr);
         feature_normalize_map[feature_set_id].push_back(max_ec);
         feature_normalize_map[feature_set_id].push_back(min_ec);
         feature_normalize_map[feature_set_id].push_back(max_coec);
         feature_normalize_map[feature_set_id].push_back(min_coec);
  
     }
     return 0;
  }
  explicit CTRParser(InputSplit *source,
                     const std::map<std::string, std::string>& args,
                     int nthread)
      : TextParserBase<IndexType>(source, nthread),threadnum_(nthread), mutex_(false){
    param_.Init(args);
    CHECK_EQ(param_.format, "ctr");
    int ret = load_feature_index("/search/odin/mxnet_add/example/ctr/feature_index");
    if (ret < 0){
        LOG(INFO) << "loading feture_index failed";
    }
    ret = load_feature_normalize("/search/odin/mxnet_add/example/ctr/feature_normalize");
       if (ret < 0){
        LOG(INFO) << "loading normalize failed";
    }
  }
  void set_threadnum(int num)
  {   
    threadnum_ = num;
  }
 protected:
  virtual void ParseBlock(char *begin,
                          char *end,
                          RowBlockContainer<IndexType> *out);

 private:
  CTRParserParam param_;
  int threadnum_;
  bool mutex_;
};

// $2 label     $4 featruelist
template <typename IndexType>
void CTRParser<IndexType>::
ParseBlock(char *begin,
           char *end,
           RowBlockContainer<IndexType> *out) {
  out->Clear();
  char * lbegin = begin;
  char * lend = lbegin;
  while (lbegin != end) {
    // get line end
    lend = lbegin + 1;
    while (lend != end && *lend != '\n' && *lend != '\r') ++lend;

    IndexType idx = 0;
    float label = 0.0f;
    
    uint32_t field_num = 0;
    char * field_list[200];
    std::string line(lbegin, lend - lbegin);
    //fprintf(stdout, "wzl %s\n", line.c_str());
    split_string('\t', (char*)line.c_str(), sizeof(field_list) / sizeof(field_list[0]), field_list, &field_num);
    if (field_num < 5) {
        // novalid line
        continue;
    }
    label = atof(field_list[2]);

    char * inner_field_list[512];
    split_string(',', field_list[5], sizeof(inner_field_list) / sizeof(inner_field_list[0]), inner_field_list, &field_num);
    for (uint32_t j = 0; j < field_num; j++) {
      out->value.push_back(atof(inner_field_list[j]));
      out->index.push_back(idx++);
    }
    // skip empty line
    while ((*lend == '\n' || *lend == '\r') && lend != end) ++lend;
    lbegin = lend;
    out->label.push_back(label);
    out->offset.push_back(out->index.size());
  }
  CHECK(out->label.size() + 1 == out->offset.size());
}

}  // namespace data
}  // namespace dmlc
#endif  // DMLC_DATA_CTR_PARSER_H_
