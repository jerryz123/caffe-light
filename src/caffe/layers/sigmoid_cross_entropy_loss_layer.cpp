#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // loss weights..
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int c = bottom[0]->channels();
  const int h = bottom[0]->height();
  const int w = bottom[0]->width();
  loss_weights_.Reshape(num,c,h,w);
  Dtype* weight = loss_weights_.mutable_cpu_data();
  vector<Dtype> tmp_w;
  if( this->layer_param_.loss_param().has_weight_source() ) {
	LOG(INFO) << "Opening file " << this->layer_param_.loss_param().weight_source();
    std::fstream infile(this->layer_param_.loss_param().weight_source().c_str(), std::fstream::in);
    CHECK(infile.is_open());
    Dtype tmp_val;
    while (infile >> tmp_val) {
      CHECK_GE(tmp_val, 0) << "Weights cannot be negative";
	  tmp_w.push_back(tmp_val);
	  LOG(INFO) << tmp_val ;
    }
    infile.close();    
    CHECK_EQ(tmp_w.size(), c);
  }
  else {
	LOG(INFO) << "NO WEIGHT SOURCE";
	tmp_w.assign(c, Dtype(1));
  }
  // make weight vector
  int w_ind;
  for (int i = 0; i < count; ++i) {
    w_ind = i % c;
	weight[i] = tmp_w[w_ind];
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* weight = loss_weights_.cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
	//LOG(INFO) << i << " : " << weight[i] ;
    loss -= weight[i] * ( input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))) );
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* weight = loss_weights_.cpu_data();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
	caffe_mul(count, weight, bottom_diff, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
