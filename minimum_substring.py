import time

# s = "abacaba"
# ("a","ba","cab","a") or ("ab","a","ca","ba")

# abacaba
# ab
# ac
# ab
# a

class Solution:
    def search(self, strs: str) -> int:
        see = []
        count = 0
        results = []
        print(strs)
        for i, x in enumerate(strs):
            if x not in see:
                see.append(x)
            else:
                count += 1
                results.append("".join(see))
                see.clear()
                see.append(x)
        # 可以預覽結果
        results.append("".join(see))
        print("results:", results)
        return count + 1


do_obj =  Solution()
s = "abacaba"
answer = do_obj.search(s)
print(s, ':', answer)
s = "aaaaaa"
answer = do_obj.search(s)
print(s, ':', answer)
