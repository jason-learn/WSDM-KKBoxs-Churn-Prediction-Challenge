# WSDM-KKBox-s-Churn-Prediction-Challenge
The 11th ACM International Conference on Web Search and Data Mining (WSDM 2018) is challenging you to build an algorithm that predicts whether a subscription user will churn using a donated dataset from KKBOX.

# Final:  rank 43/575

userlog_features分两个角度：过往所有时间段的features | 过往部分时间段的features <br>

process_userlog_feb.py 提取二月份训练数据的features <br>
process_userlog_mar.py 提取三月份测试数据的features <br>
process_userlog_all.py 提取过往所有时间段的features <br>

process_features_userlog_feb_mar.py 提取过往一个月的交叉features <br>
process_features_userlog_all.py     提取过往所有时间段的交叉features <br>
