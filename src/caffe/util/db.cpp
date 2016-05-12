#include "caffe/util/db.hpp"

#include <string>

namespace caffe { namespace db {

DB* GetDB(DataParameter::DB backend) {
  // Disable DB
  LOG(FATAL) << "Unknown database backend";
}

DB* GetDB(const string& backend) {
  // Disable DB
  LOG(FATAL) << "Unknown database backend";
}

}  // namespace db
}  // namespace caffe
