'''
49
'''
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hash_table = {}
        for word in strs:
            word_key = ''.join(sorted(word))
            hash_table.setdefault(word_key, []).append(word)
        return [value for _, value in hash_table.items()]
    
'''
128
'''
import collections
class UF:
    def __init__(self, nums):
        self.parents = {num: num for num in nums}
        self.cnts = collections.defaultdict(lambda:1)
        
    def find(self, kid:int) -> int:
        while kid != self.parents[kid]:
            kid = self.parents[kid]
        return kid

    def union(self, pre, post):
        if post not in self.parents:
            return 1
        root_pre, root_post = self.find(pre), self.find(post)
        if root_pre == root_post:
            return self.cnts[root_pre]
        self.parents[root_post] = root_pre
        self.cnts[root_pre] += self.cnts[root_post]
        return self.cnts[root_pre]

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if len(nums)==0:return 0
        uf = UF(nums)
        res = 1
        for num in nums:
            res = max(res, uf.union(num, num+1))
        return res
    
'''
283
'''
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        low = 0
        n = len(nums)
        for fast in range(n):
            if nums[fast] != 0:
                nums[low] = nums[fast]
                low += 1
        for idx in range(low, n):
            nums[idx] = 0

'''
11
'''
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            area = min(height[l], height[r]) * (r - l)
            ans = max(ans, area)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return ans
    
'''
15
'''
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        No_repeat = set()
        res = []
        for i in range(n):
            L=i+1
            R=n-1
            while(L<R):
                if(nums[i]+nums[L]+nums[R] == 0):
                    key = (nums[i], nums[L], nums[R])
                    if key not in No_repeat:
                        No_repeat.add(key)
                        res.append([nums[i],nums[L],nums[R]])
                    else:
                        pass
                    L = L + 1
                    R = R - 1
                elif(nums[i]+nums[L]+nums[R] > 0):
                    R = R - 1
                else:
                    L = L + 1
        return res

        
'''
16
'''
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        diff = float('inf')
        n = len(nums)
        res = 0
        for idx in range(n-2):
            left = idx + 1
            right = n-1
            while(left != right):
                sums = nums[idx] + nums[left] + nums[right]
                new_diff = sums - target
                abs_new_diff = abs(new_diff)
                if new_diff == 0:
                    return sums
                elif new_diff > 0:
                    right -= 1
                else:
                    left += 1
                if abs_new_diff < diff:
                    diff = abs_new_diff
                    res = sums
        return res

'''
17
'''
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:return []
        num_dict = {'2':['a','b','c'],'3':['d','e','f'],'4':['g','h','i'],'5':['j','k','l'],
                    '6':['m','n','o'],'7':['p','q','r','s'],'8':['t','u','v'],'9':['w','x','y','z']}
        combinations = []
        n = len(digits)
        def backtrack(idx:int, path:str):
            if idx == n:
                combinations.append(path)
                return

            for letter in num_dict[digits[idx]]:
                backtrack(idx+1, path+letter)

        backtrack(0, "")
        return combinations

'''
42
'''
class Solution:
    def trap(self, height: List[int]) -> int:
        # 初始化答案和双指针
        ans = left = pre_max = suf_max = 0
        right = len(height) - 1

        # 当左指针不超过右指针时，循环继续
        while left <= right:
            # 更新左边的最高柱子
            pre_max = max(pre_max, height[left])
            # 更新右边的最高柱子
            suf_max = max(suf_max, height[right])

            # 如果左边的最高柱子低于右边的最高柱子
            if pre_max < suf_max:
                # 计算当前位置能接收的雨水量，并移动左指针
                ans += pre_max - height[left]
                left += 1
            else:
                # 否则，计算当前位置能接收的雨水量，并移动右指针
                ans += suf_max - height[right]
                right -= 1

        # 返回计算出的总雨水量
        return ans

'''
3
'''
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = left = 0
        window = set()
        for right, c in enumerate(s):
            while c in window:
                window.remove(s[left])
                left += 1
            window.add(c)
            ans = max(ans, right - left + 1)
        return ans

'''
438
'''
from collections import Counter

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        # 如果p长度大于s，直接返回空列表
        if len(p) > len(s):
            return []

        # 初始化哈希表
        p_count = Counter(p)  # 字符串p的字符计数
        s_count = Counter()  # 字符串s的滑动窗口字符计数

        result = []
        # 遍历字符串s
        for i in range(len(s)):
            # 将当前字符添加到s的计数器
            s_count[s[i]] += 1
            # 如果窗口大小超过p的长度，则从左边移除一个字符
            if i >= len(p):
                if s_count[s[i - len(p)]] == 1:
                    del s_count[s[i - len(p)]]
                else:
                    s_count[s[i - len(p)]] -= 1
            # 如果两个计数器相等，则找到一个起始索引
            if p_count == s_count:
                result.append(i - len(p) + 1)

        return result
