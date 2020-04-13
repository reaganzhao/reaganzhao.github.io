---
layout: projects
title: Phishing Detection
mathjax: true
---

School project with Python and scikit-learn

<div style="text-align: center"><img src="/images/phishing.jpg" width="250" /></div>

The purpose of this project is to predict phishing website using [Mendeley Phishing Dataset](https://data.mendeley.com/datasets/h3cgnj8hft/1), and compare different classification models. There have been famous detection problems such as Credit card Fraud Detection, while people have not done great phishing detection because they don't have data with enough attributes. This data collects the Selenium browser network and contains 48 related features. We finally proved that these features are important and good for classification accuracy.

Main packages and version: scikit-learn==0.21.3 pandas==0.25.1 rfpimp==1.3.4

We used scikit-learn classification models such as logistic regression, random forest, support vector machine, KNN Classifiers as well as Gradient Boosting. The best model comes with Gradient boosting estimators=500, max_depth = 5 and learning rate=0.5.

Thanks to Prof. [Brian Spiering](https://github.com/brianspiering) for his guidance and support on this school project. Also, thanks to my teammates Yao Liu and Daren Ma for their contribution.