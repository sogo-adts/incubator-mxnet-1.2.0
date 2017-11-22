/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_ctr.cc
 * \brief define a CTR Reader to read in arrays
 */
#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/data.h>
#include "./iter_prefetcher.h"
#include "./iter_batchloader.h"

namespace mxnet {
namespace io {
// CTR parameters
struct CTRIterParam : public dmlc::Parameter<CTRIterParam> {
  /*! \brief path to data ctr file */
  std::string data_ctr;
  /*! \brief data shape */
  TShape data_shape;
  /*! \brief path to label ctr file */
  std::string label_ctr;
  /*! \brief label shape */
  TShape label_shape;
  bool shuffle;
  /*! \brief partition the data into multiple parts */
  int num_parts;
  /*! \brief the index of the part will read*/
  int part_index;
  // declare parameters
  DMLC_DECLARE_PARAMETER(CTRIterParam) {
    DMLC_DECLARE_FIELD(data_ctr)
        .describe("Dataset Param: Data ctr path.");
    DMLC_DECLARE_FIELD(data_shape)
        .describe("Dataset Param: Shape of the data.");
    index_t shape1[] = {1};
    DMLC_DECLARE_FIELD(label_shape).set_default(TShape(shape1, shape1 + 1))
        .describe("Dataset Param: Shape of the label.");
     DMLC_DECLARE_FIELD(shuffle).set_default(true)
        .describe("Augmentation Param: Whether to shuffle data.");
    DMLC_DECLARE_FIELD(num_parts).set_default(1)
        .describe("partition the data into multiple parts");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("the index of the part will read");
  }
};

class CTRIter: public IIterator<DataInst> {
 public:
  CTRIter() {
    out_.data.resize(2);
    srand((unsigned)time(NULL));
  }
  virtual ~CTRIter() {}

  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      param_.InitAllowUnknown(kwargs);
      data_parser_.reset(dmlc::Parser<uint32_t>::Create(param_.data_ctr.c_str(), param_.part_index, param_.num_parts, "ctr"));
  }

  virtual void BeforeFirst() {
    data_parser_->BeforeFirst();
    data_ptr_ = 0;
    data_size_ = 0;
    inst_counter_ = 0;
    end_ = false;
  }

  virtual bool Next() {
    if (end_) return false;
    while (data_ptr_ >= data_size_) {
      if (!data_parser_->Next()) {
        end_ = true; return false;
      }
      data_ptr_ = 0;
      data_size_ = data_parser_->Value().size;
    }
    out_.index = inst_counter_++;
    CHECK_LT(data_ptr_, data_size_);
    out_.data[0] = AsTBlob(data_parser_->Value()[data_ptr_], param_.data_shape);
    out_.data[1] = AsTBlob(data_parser_->Value()[data_ptr_++], param_.label_shape, true);
    return true;
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

 private:
  inline TBlob AsTBlob(const dmlc::Row<uint32_t>& row, const TShape& shape, bool is_label = false) {
    const real_t* ptr = 0;
    if (is_label) {
        ptr = row.label;
    }
    else {
        CHECK_EQ(row.length, shape.Size())
            << "The data size in CTR do not match size of shape: "
            << "specified shape=" << shape << ", the ctr row-length=" << row.length;
        ptr = row.value;
    }
    return TBlob((real_t*)ptr, shape, cpu::kDevMask);  // NOLINT(*)
  }

  CTRIterParam param_;
  // output instance
  DataInst out_;
  // internal instance counter
  unsigned inst_counter_{0};
  // at end
  bool end_{false};
  // label parser
  volatile size_t data_ptr_{0}, data_size_{0};
  // label parser
  std::unique_ptr<dmlc::Parser<uint32_t> > data_parser_;
};


DMLC_REGISTER_PARAMETER(CTRIterParam);

MXNET_REGISTER_IO_ITER(CTRIter)
.describe("Create iterator for dataset in ctr.")
.add_arguments(CTRIterParam::__FIELDS__())
.set_body([]() {
    return new PrefetcherIter(
        new BatchLoader(
            new CTRIter()));
  });

}  // namespace io
}  // namespace mxnet
