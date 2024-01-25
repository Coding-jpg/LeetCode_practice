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

        
        