前缀树是用来快速检索的树结构
建立一棵多叉树，通过字符串的公共前缀来降低查询时间的开销，是一种空间换时间的策略

基本操作包括三种：查找，插入，删除

每棵树结点，除了可以存放是否存在以当前字为结尾的词，还可以存储其他信息：
    比如如果我们把词典构造为一棵前缀树，那么每个结点还可以存放该词在词典中的索引，当查找某个词时，直接返回该索引（对于没找到的可以设置一个默认值），即可快速定位

    