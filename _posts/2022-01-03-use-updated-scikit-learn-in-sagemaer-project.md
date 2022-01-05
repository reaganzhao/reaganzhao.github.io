---
layout: post
title:  Upgrade scikit-learn in sagemaker
image:
  feature: gate_crop.png
tags:   programming
date:   2022-01-03 16:40
---

Sagemaker has its own templates for machine learning model training, deployment, and monitoring. One sad thing about sagemaker is that the newest scikit-learn version it supports until now is 0.23-1, which is missing a very import feature for label encoding, dealing with missing categories and [other features](https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_0_24_0.html). There are two ways to enable the missing categories 1. Amend your label encoding class and 2.Create your own image container for new versions of scikit-learn


## 1. Redefine the encoding class

The first way to deal with the label encoding is to write your own label encoding class:
```python
class OrdinalEncoderNew(object):
    def __init__(self):

        self.ordinal_encoder = OrdinalEncoder()

    def fit(self, array, y=None):
        array = np.vstack([df_cate_test, ['unknown']*df_cate_test.shape[1]])
        self.ordinal_encoder = self.ordinal_encoder.fit(array)

        return self

    def transform(self, array, y=None):
        array_idx = len(self.ordinal_encoder.categories_)
        array_transpose=array.T
        for array_idx in range(array_idx):
            new_item_list = []
            for unique_item in np.unique(array_transpose[array_idx]):
                if unique_item not in self.ordinal_encoder.categories_[array_idx]:
                    new_item_list.append(unique_item)
            array_transpose[array_idx] = np.array(['unknown' if x in new_item_list else x for x in array_transpose[array_idx]], dtype='object')

        return self.ordinal_encoder.transform(array)
```
Your can then call OrdinalEncoderNew() in your preprocessing object. The drawback of this method is you may see a "Module Main has No Attribute" error. I think the best way to solve it is to follow this [`blog`](https://rebeccabilbro.github.io/module-main-has-no-attribute/) by creating a new file to write your class in it. The thing is after using that you will need to use "source_dir" or "dependencies" to use it for model preprocessing and training.

## 2. Create your own image container for new scikit-learn version

I prefer this method because 1.you are able to use other features in scikit-learn and 2.use it in other notebooks easily. It can be much more complicated if you need to build the image from scratch all by yourself. Fortunately, pagemaker's [`default container`](https://github.com/aws/sagemaker-scikit-learn-container) makes it easy for you to build the image on top of it.

So the steps to build the image will be 
1. Write the 0.24 image scripts, there is someone who already committed the [`code`](https://github.com/aws/sagemaker-scikit-learn-container/compare/master...tophatter:dh-v24) for it. It's not merged but you can clone and pull it for use.
2. Follow the [`instructions`](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html) to create the image and push it to ECR. You can build the image locally but I created it on EC2 simply because it's easier to set up the permissions on AWS.
3. Attach the image you built in ECR to sagemaker and use it for model training. Go to the sagemaker UI click on images on the left bar, click on "create image" and then you can attach your image URI there. After that, you are ready to use your own image!

