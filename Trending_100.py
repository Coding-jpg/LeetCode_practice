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

'''
560
'''
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        dic={0:1}
        sums,res=0,0
        for num in nums:
            sums+=num
            res+=dic.get(sums-k,0)
            dic[sums]=dic.get(sums,0)+1
        return res

'''
239
'''
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        q = deque()
        for i, x in enumerate(nums):
            while q and nums[q[-1]] <= x:
                q.pop()
            q.append(i)
            if i - q[0] >= k:
                q.popleft()
            if i >= k-1:
                res.append(nums[q[0]])

        return res

'''
76
'''
import collections
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)
        need_cnt, i = len(t), 0
        res = (0, float('inf'))

        for j, char in enumerate(s):
            if need[char] > 0:
                need_cnt -= 1
            need[char] -= 1

            if need_cnt == 0:
                while i < j and need[s[i]] < 0:
                    need[s[i]] += 1
                    i += 1
                if j - i < res[1] - res[0]:
                    res = (i, j)
                need[s[i]] += 1
                need_cnt += 1
                i += 1

        return '' if res[1] == float('inf') else s[res[0]:res[1] + 1]

'''
53
'''
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        dp = [0 for _ in range(n)]
        dp[0] = nums[0]
        for i in range(1, n):
            if dp[i-1] >= 0:
                dp[i] = dp[i-1] + nums[i]
            else:
                dp[i] = nums[i]
        return max(dp)
    
'''
56
'''
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        n = len(intervals)
        intervals_s = sorted(intervals, key=lambda x:x[0])
        res = [intervals_s[0]]
        for idx in range(1,n):
            if res[-1][1] >= intervals_s[idx][1]:
                continue
            elif res[-1][1] >= intervals_s[idx][0] and res[-1][1] < intervals_s[idx][1]:
                res[-1] = [res[-1][0], intervals_s[idx][1]]
            else:
                res.append(intervals_s[idx])
        return res

'''
189
'''
from collections import deque

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        queue = deque(maxlen=len(nums))
        for num in nums:
            queue.append(num)
        queue.rotate(k)
        for idx, num in enumerate(queue):
            nums[idx] = num
        
'''
238
'''
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        s1, s2 = [1] * (n + 2), [1] * (n + 2)
        for i in range(1, n + 1):
            s1[i] = s1[i - 1] * nums[i - 1]
        for i in range(n, 0, -1):
            s2[i] = s2[i + 1] * nums[i - 1]
        ans = [s1[i - 1] * s2[i + 1] for i in range(1, n + 1)]
        return ans
    
'''
41
'''
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        return min(set(range(1, len(nums) + 2)) - set(nums))
    
'''
73
'''
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        row, col = [False] * m, [False] * n 

        for r in range(m):
            for c in range(n):
                if matrix[r][c] == 0:
                    row[r] = col[c] = True
        
        for r in range(m):
            for c in range(n):
                if row[r] or col[c]:
                    matrix[r][c] = 0

'''
48
'''
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for i in range(n // 2):
            for j in range((n + 1) // 2):
                tmp = matrix[i][j]
                matrix[i][j] = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = tmp

'''
240
'''
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        i, j = len(matrix) - 1, 0
        while i >= 0 and j < len(matrix[0]):
            if matrix[i][j] > target: i -= 1
            elif matrix[i][j] < target: j += 1
            else: return True
        return False

'''
160
'''
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
    
'''
206
'''
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur, pre = head, None
        while cur:
            cur.next, pre, cur = pre, cur, cur.next
        return pre
    
'''
234
'''
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if not head:
            return True
        
        # 快慢指针找中间节点
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # 反转后半部分链表
        prev = None
        while slow:
            tmp = slow.next
            slow.next = prev
            prev = slow
            slow = tmp
        
        # 比较前半部分和反转后的后半部分
        left, right = head, prev
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        
        return True

# 测试代码
# 创建一个示例链表 1 -> 2 -> 2 -> 1
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(2)
head.next.next.next = ListNode(1)

# 检查链表是否为回文
sol = Solution()
sol.isPalindrome(head)

'''
141
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        hash_map = set()
        pre = head.next
        while pre:
            if pre not in hash_map:
                hash_map.add(pre)
                pre = pre.next
            else:
                return True
        return False
            
        