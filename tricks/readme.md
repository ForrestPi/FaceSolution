https://github.com/themostnewone/2019ccf 
https://discussion.datafountain.cn/questions/2244?new=0

https://discussion.datafountain.cn/questions/1904/answers/22795
Trick1: 水平翻转的增广,增广方法已包含在最原始版本baseline中
Trick2: 重新检测+对齐人脸,会使模型表现有显著提升
Trick3: 多个开源模型结果融合与TTA
Trick4: 输入图片做预处理,如直方图均衡化等,也可以作为tta的一部分
Trick5: 把余弦相似度合理映射到[0,1]区间,比如lambda x: (x+0.3)/1.3
可能有用的idea: 使用训练集训练人种分类器,对开源数据集进行人种分类;使用分类后的4个人种数据加上训练集数据,分别针对每个人种重新训练arcface(insight face). 同时在训练时可以增加更多颜色方面的增广.