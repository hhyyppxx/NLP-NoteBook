

class TreeNode():
    def __init__(self) -> None:
        self.end = 0
        self.child = dict()

class Trie():
    def __init__(self) -> None:
        self.root = TreeNode()
    
    def insert(self, word):
        if not word:
            return 
        
        node = self.root
        for ch in word:
            if ch not in node.child:
                node.child[ch] = TreeNode()
            node = node.child[ch]

        node.end += 1   # 当前字为某个词的结尾，记录

    def search(self, word):
        if not word:
            return False
        
        node = self.root
        for ch in word:
            if ch not in node.child:
                return False
            
            node = node.child[ch]
        
        if node.end > 0:    #   必须得有以当前字为结尾的词才算查到
            return True
        
        return False

if __name__ == "__main__":

    word_dict = ['一', '一二', '一二三', '一二五', '一三五', '二五', '二三', '五三', '五五']
    pre_tree = Trie()
    for word in word_dict:
        pre_tree.insert(word)
    

    print('一三五 in word_dict ? ', pre_tree.search('一三五'))
    print('一三四 in word_dict ? ', pre_tree.search('一三四'))
    print('一五 in word_dict ? ', pre_tree.search('一五'))
    print('五二 in word_dict ? ', pre_tree.search('五二'))
    print('五二 in word_dict ? ', pre_tree.search('五五'))
