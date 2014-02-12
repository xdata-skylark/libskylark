#ifndef SKYLARK_FEATURE_TRANSFORM_TAGS
#define SKYLARK_FEATURE_TRANSFORM_TAGS

namespace skylark { namespace ml {

struct feature_transform_type_tag { } ;

struct regular_feature_transform_tag : public feature_transform_type_tag { } ;

struct fast_feature_transform_tag : public feature_transform_type_tag { } ;

} } 
#endif // SKYLARK_FEATURE_TRANSFORM_TAGS
