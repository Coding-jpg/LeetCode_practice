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
            
'''
142
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:return None
        hash_map = set()
        cur = head
        while cur:
            if cur not in hash_map:
                hash_map.add(cur)
                cur = cur.next
            else:
                return cur
        return None
        
'''
21
'''
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        if not list1: return list2
        if not list2: return list1
        head = ListNode()
        cur = head
        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        if not list1:
            cur.next = list2
        elif not list2:
            cur.next = list1
        return head.next
        
'''
2
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode], carry=0) -> Optional[ListNode]:
        cur = dummy = ListNode()
        carry = 0
        while l1 or l2 or carry:
            carry += (l1.val if l1 else 0) + (l2.val if l2 else 0)
            cur.next = ListNode(carry % 10)
            carry //= 10
            if l1: l1 = l1.next
            if l2: l2 = l2.next
        return dummy.next
        
'''
19
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        left = right = dummy = ListNode(next=head)
        for _ in range(n):
            right = right.next
        while right.next:
            left = left.next
            right = right.next
        left.next = left.next.next
        return dummy.next
    
'''
24
'''
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = dummy = ListNode(next=head)
        if not prev.next or not prev.next.next: return head
        first, second = head, head.next
        flag = 2
        while prev and first and second:
            if flag == 2:
                prev.next = second
                first.next = second.next
                second.next = first
                flag = 0

            prev = prev.next
            first = prev.next
            second = first.next
            flag += 1
        return dummy.next
            
'''
25
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        n = 0
        cur = head
        while cur:
            n += 1  # 统计节点个数
            cur = cur.next

        p0 = dummy = ListNode(next=head)
        pre = None
        cur = head
        while n >= k:
            n -= k
            for _ in range(k): 
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt

            nxt = p0.next
            nxt.next = cur
            p0.next = pre
            p0 = nxt
        return dummy.next
    
'''
138
'''
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return
        dic = {}
        # 3. 复制各节点，并建立 “原节点 -> 新节点” 的 Map 映射
        cur = head
        while cur:
            dic[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        # 4. 构建新节点的 next 和 random 指向
        while cur:
            dic[cur].next = dic.get(cur.next)
            dic[cur].random = dic.get(cur.random)
            cur = cur.next
        # 5. 返回新链表的头节点
        return dic[head]

'''
148
'''
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        h, length, intv = head, 0, 1
        while h: h, length = h.next, length + 1
        res = ListNode(0)
        res.next = head
        # merge the list in different intv.
        while intv < length:
            pre, h = res, res.next
            while h:
                # get the two merge head `h1`, `h2`
                h1, i = h, intv
                while i and h: h, i = h.next, i - 1
                if i: break # no need to merge because the `h2` is None.
                h2, i = h, intv
                while i and h: h, i = h.next, i - 1
                c1, c2 = intv, intv - i # the `c2`: length of `h2` can be small than the `intv`.
                # merge the `h1` and `h2`.
                while c1 and c2:
                    if h1.val < h2.val: pre.next, h1, c1 = h1, h1.next, c1 - 1
                    else: pre.next, h2, c2 = h2, h2.next, c2 - 1
                    pre = pre.next
                pre.next = h1 if c1 else h2
                while c1 > 0 or c2 > 0: pre, c1, c2 = pre.next, c1 - 1, c2 - 1
                pre.next = h 
            intv *= 2
        return res.next

'''
23
'''
ListNode.__lt__ = lambda a, b: a.val < b.val  # 让堆可以比较节点大小

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        cur = dummy = ListNode()  # 哨兵节点，作为合并后链表头节点的前一个节点
        h = [head for head in lists if head]  # 初始把所有链表的头节点入堆
        heapify(h)  # 堆化
        while h:  # 循环直到堆为空
            node = heappop(h)  # 剩余节点中的最小节点
            if node.next:  # 下一个节点不为空
                heappush(h, node.next)  # 下一个节点有可能是最小节点，入堆
            cur.next = node  # 合并到新链表中
            cur = cur.next  # 准备合并下一个节点
        return dummy.next  # 哨兵节点的下一个节点就是新链表的头节点

'''
146
'''
from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

'''
94
'''
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

'''
104
'''
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        return max(left_depth, right_depth) + 1
    
'''
226
'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def invertTraversal(root):
    if root is None:
        return
    root.left, root.right = root.right, root.left
    invertTraversal(root.left)
    invertTraversal(root.right)

class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        invertTraversal(root)
        return root
    
'''
101
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque

class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root is None:
            return False
        queue_left = deque([root.left])
        queue_right = deque([root.right])
        while queue_left and queue_right:
            node_left = queue_left.popleft()
            node_right = queue_right.popleft()
            
            if node_left is None and node_right is None:
                continue
            elif node_left is None or node_right is None:
                return False
            elif node_left.val != node_right.val:
                return False

            queue_left.append(node_left.left)
            queue_left.append(node_left.right)
            queue_right.append(node_right.right)
            queue_right.append(node_right.left)
        
        return not queue_left and not queue_right
    
'''
543
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        ans = 0
        def dfs(node: Optional[TreeNode]) -> int:
            if node is None:
                return -1
            l_len = dfs(node.left) + 1
            r_len = dfs(node.right) + 1
            nonlocal ans
            ans = max(ans, l_len + r_len)
            return max(l_len, r_len)
        dfs(root)
        return ans
    
'''
102
'''
from collections import deque

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None: return []
        level_queque = deque([root])
        res = []
        while len(level_queque) != 0:
            vals = []
            for _ in range(len(level_queque)):
                cur = level_queque.popleft()
                vals.append(cur.val)
                if cur.left: level_queque.append(cur.left)
                if cur.right: level_queque.append(cur.right)
            res.append(vals)
        return res
    
'''
108
'''
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        return self.build(nums, 0, len(nums)-1)
        
    def build(self, nums, l, r):
        if l > r:
            return None
        mid = l + r >> 1
        ans = TreeNode(nums[mid])
        ans.left = self.build(nums, l, mid-1)
        ans.right = self.build(nums, mid+1, r)
        return ans
    
'''
98
'''
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        self.prev = None
        return self.inorderTraversal(root)
        
    def inorderTraversal(self, root: Optional[TreeNode]) -> bool:
        if root is None: 
            return True
        if not self.inorderTraversal(root.left):
            return False
        if self.prev is not None and self.prev.val >= root.val:
            return False
        self.prev = root
        return self.inorderTraversal(root.right)
    
'''
230
'''
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.cnt = 0
        self.result = None
        self.inorderTraversal(root, k)
        return self.result

    def inorderTraversal(self, root: Optional[TreeNode], k: int):
        if root is None:
            return
        self.inorderTraversal(root.left, k)
        self.cnt += 1
        if self.cnt == k:
            self.result = root.val
            return
        self.inorderTraversal(root.right, k)

'''
199
'''
from collections import deque

class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        level_queue = deque([root])
        res = []
        while level_queue:
            level_list = []
            for _ in range(len(level_queue)):
                cur = level_queue.popleft()
                level_list.append(cur.val)
                if cur.left: level_queue.append(cur.left)
                if cur.right: level_queue.append(cur.right)
            res.append(level_list[-1])
        return res    

'''
114
'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
from collections import deque

class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.TreeNode_queue = deque()
        self.preorder_Traversal(root)
        dummy = TreeNode(-1)
        cur = dummy
        while self.TreeNode_queue:
            cur.right = self.TreeNode_queue.popleft()
            cur.left = None
            cur = cur.right
        root = dummy.right
        
    def preorder_Traversal(self, node: Optional[TreeNode]) -> None:
        if node is None: return
        self.TreeNode_queue.append(node)
        self.preorder_Traversal(node.left)
        self.preorder_Traversal(node.right)

'''
105
'''
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def recur(root, left, right):
            if left > right: return                               # 递归终止
            node = TreeNode(preorder[root])                       # 建立根节点
            i = dic[preorder[root]]                               # 划分根节点、左子树、右子树
            node.left = recur(root + 1, left, i - 1)              # 开启左子树递归
            node.right = recur(i - left + root + 1, i + 1, right) # 开启右子树递归
            return node                                           # 回溯返回根节点

        dic, preorder = {}, preorder
        for i in range(len(inorder)):
            dic[inorder[i]] = i
        return recur(0, 0, len(inorder) - 1)

'''
437
'''
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        self.cnt = 0
        self.dfs(root, targetSum, [])
        return self.cnt

    def dfs(self, node, target, path_sum):
        if not node:
            return

        # 更新当前路径和
        path_sum = [num + node.val for num in path_sum] + [node.val]

        # 计算等于目标和的路径数
        self.cnt += path_sum.count(target)

        # 递归遍历左右子树
        self.dfs(node.left, target, path_sum)
        self.dfs(node.right, target, path_sum)

'''
236
'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None or root == p or root == q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left is not None and right is not None:
            return root
        elif left is not None:
            return left
        else:
            return right

'''
124
'''
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')
        self.max_gain(root)
        return self.max_sum

    def max_gain(self, node: Optional[TreeNode]) -> int:
        if not node:
            return 0

        # 计算左右子树的最大贡献，如果贡献为负则忽略
        left_gain = max(self.max_gain(node.left), 0)
        right_gain = max(self.max_gain(node.right), 0)

        # 当前节点的最大路径和
        price_newpath = node.val + left_gain + right_gain

        # 更新全局最大路径和
        self.max_sum = max(self.max_sum, price_newpath)

        # 返回节点的最大贡献值
        return node.val + max(left_gain, right_gain)

'''
200
'''
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        res = 0

        def dfs(x, y):
            if grid[x][y] == '1':
                grid[x][y] = '0'
            else:
                return
            if x > 0:
                dfs(x - 1, y)
            if x < m - 1:
                dfs(x + 1, y)
            if y > 0:
                dfs(x, y - 1)
            if y < n - 1:
                dfs(x, y + 1)
            
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    dfs(i, j)
                    res += 1
        return res

'''
994
'''
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        rot_queue = deque()
        fresh_count = 0

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2:
                    rot_queue.append((i, j))
                elif grid[i][j] == 1:
                    fresh_count += 1

        minutes_passed = 0
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while rot_queue and fresh_count:
            minutes_passed += 1
            for _ in range(len(rot_queue)): 
                x, y = rot_queue.popleft()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                        grid[nx][ny] = 2
                        fresh_count -= 1
                        rot_queue.append((nx, ny))

        return minutes_passed if fresh_count == 0 else -1

'''
207
'''
from collections import deque

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegrees = [0 for _ in range(numCourses)]
        adjacency = [[] for _ in range(numCourses)]
        queue = deque()
        # Get the indegree and adjacency of every course.
        for cur, pre in prerequisites:
            indegrees[cur] += 1
            adjacency[pre].append(cur)
        # Get all the courses with the indegree of 0.
        for i in range(len(indegrees)):
            if not indegrees[i]: queue.append(i)
        # BFS TopSort.
        while queue:
            pre = queue.popleft()
            numCourses -= 1
            for cur in adjacency[pre]:
                indegrees[cur] -= 1
                if not indegrees[cur]: queue.append(cur)
        return not numCourses

'''
208
'''
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for s in word:
            node = node.children[s]
        node.is_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for s in word:
            if s in node.children:
                node = node.children[s]
            else:
                return False
        return node.is_word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for s in prefix:
            if s in node.children:
                node = node.children[s]
            else:
                return False
        return True

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

'''
46
'''
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(x):
            if x == len(nums) - 1:
                res.append(list(nums))
                return
            for i in range(x, len(nums)):
                nums[i], nums[x] = nums[x], nums[i]
                dfs(x+1)
                nums[i], nums[x] = nums[x], nums[i]

        res = []
        dfs(0)
        return res
    
'''
78
'''
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        
        def helper(i, tmp):
            res.append(tmp)
            for j in range(i, n):
                helper(j + 1,tmp + [nums[j]] )
        helper(0, [])
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
39
'''
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(
            state, target, choices, start, res
        ):
            if target == 0:
                res.append(list(state))
                return
            for i in range(start, len(choices)):
                if target - choices[i] < 0:
                    break
                state.append(choices[i])
                backtrack(state, target - choices[i], choices, i, res)
                state.pop()
        state = []
        candidates.sort()
        start = 0
        res = []
        backtrack(state, target, candidates, start, res)
        return res

'''
22
'''

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        cur_str = ''

        def dfs(cur_str, left, right, n):   
            if left == n and right == n:
                res.append(cur_str)
                return 
            elif left > n or right > n:
                return  
            if left < right:
                return  
            
            if left < n:
                dfs(cur_str+'(', left+1, right, n)
            if right < n:
                dfs(cur_str+')', left, right+1, n)
        dfs(cur_str, 0, 0, n)
        return res

'''
79
'''
from typing import List

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        rows, cols = len(board), len(board[0])
        n = len(word)
        
        def dfs(idx, row, col, visited):
            if idx == n:
                return True
            if row < 0 or col < 0 or row >= rows or col >= cols:
                return False
            if visited[row][col] or board[row][col] != word[idx]:
                return False
            
            visited[row][col] = True
            found = (dfs(idx + 1, row + 1, col, visited) or
                     dfs(idx + 1, row - 1, col, visited) or
                     dfs(idx + 1, row, col + 1, visited) or
                     dfs(idx + 1, row, col - 1, visited))
            visited[row][col] = False
            return found

        visited = [[False for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if dfs(0, r, c, visited):
                    return True
        return False

'''
131
'''
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        ans = []
        path = []
        n = len(s)
        def dfs(i: int) -> None:
            if i == n:
                ans.append(path.copy())
                return
            for j in range(i, len(s)):
                t = s[i: j+1]
                if t == t[::-1]:
                    path.append(t)
                    dfs(j+1)
                    path.pop()
        dfs(0)
        return ans

'''
51
'''
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def is_safe(board, row, col, N):
            for i in range(row):
                if board[i][col] == 'Q':
                    return False
            for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
                if board[i][j] == 'Q':
                    return False
            for i, j in zip(range(row, -1, -1), range(col, N)):
                if board[i][j] == 'Q':
                    return False
            return True
        
        def solve(board, row, N):
            if row == N:
                solutions.append(["".join(row) for row in board])

            for col in range(N):
                if is_safe(board, row, col, N):
                    board[row][col] = 'Q'
                    solve(board, row+1, N)
                    board[row][col] = '.'
        
        solutions = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        solve(board, 0, n)
        return solutions
    
'''
35
'''
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        def search(sub_list: List[int], left: int, right: int):
            if left > right:
                return left  # 如果找不到，返回应该插入的位置

            mid = (left + right) // 2
            if target == sub_list[mid]:
                return mid
            elif target < sub_list[mid]:
                return search(sub_list, left, mid - 1)
            else:
                return search(sub_list, mid + 1, right)

        return search(nums, 0, len(nums) - 1)

'''
74
'''
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        nums_list = []
        for nums in matrix:
            nums_list.extend(nums)
        def search(nums:List[int], left:int, right:int) -> bool:
            if left > right:
                return False
            mid = (left + right) // 2
            if target == nums[mid]:
                return True
            elif target < nums[mid]:
                return search(nums, left, mid-1)
            else:
                return search(nums, mid+1, right)
        return search(nums_list, 0, len(nums_list)-1)
    
'''
34
'''
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def findLeftMost(nums: List[int], target: int) -> int:
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left

        def findRightMost(nums: List[int], target: int) -> int:
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] <= target:
                    left = mid + 1
                else:
                    right = mid - 1
            return right

        left, right = findLeftMost(nums, target), findRightMost(nums, target)
        if left <= right:
            return [left, right]
        else:
            return [-1, -1]

'''
33
'''
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1

        # 处理只有一个元素的情况
        if len(nums) == 1:
            return 0 if nums[0] == target else -1

        max_idx = nums.index(max(nums))
        sorted_nums = nums[max_idx+1:] + nums[:max_idx+1]

        def bi_search(sorted_nums: List[int], left: int, right: int) -> int:
            if left > right:
                return -1
            mid = (left + right) // 2
            if target == sorted_nums[mid]:
                return (mid + max_idx + 1) % len(nums)
            elif target < sorted_nums[mid]:
                return bi_search(sorted_nums, left, mid - 1)
            else:
                return bi_search(sorted_nums, mid + 1, right)

        return bi_search(sorted_nums, 0, len(sorted_nums) - 1)

'''
153
'''
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < nums[right]:
                right = mid
            else:
                left = mid + 1
        return nums[left]

'''
4
'''
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # 确保 nums1 是较短的数组
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        m, n = len(nums1), len(nums2)
        imin, imax, half_len = 0, m, (m + n + 1) // 2
        
        while imin <= imax:
            i = (imin + imax) // 2
            j = half_len - i
            if i < m and nums2[j-1] > nums1[i]:
                # i 太小，需要增大
                imin = i + 1
            elif i > 0 and nums1[i-1] > nums2[j]:
                # i 太大，需要减小
                imax = i - 1
            else:
                # i 正好
                if i == 0: max_of_left = nums2[j-1]
                elif j == 0: max_of_left = nums1[i-1]
                else: max_of_left = max(nums1[i-1], nums2[j-1])

                if (m + n) % 2 == 1:
                    return max_of_left

                if i == m: min_of_right = nums2[j]
                elif j == n: min_of_right = nums1[i]
                else: min_of_right = min(nums1[i], nums2[j])

                return (max_of_left + min_of_right) / 2

        return 0.0

'''
20
'''
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {')': '(', ']': '[', '}': '{'}
        for char in s:
            if char in mapping:
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:
                stack.append(char)
        return not stack

'''
155
'''
class MinStack:

    def __init__(self):
        self.content = []

    def push(self, val: int) -> None:
        self.content.append(val)

    def pop(self) -> None:
        if self.content:
            self.content.pop()

    def top(self) -> int:
        return self.content[-1]

    def getMin(self) -> int:
        return min(self.content)

'''
394
'''
class Solution:
    def decodeString(self, s: str) -> str:
        stack, res, multi = [], "", 0
        for c in s:
            if c == '[':
                stack.append([multi, res])
                res, multi = "", 0
            elif c == ']':
                cur_multi, last_res = stack.pop()
                res = last_res + cur_multi * res
            elif '0' <= c <= '9':
                multi = multi * 10 + int(c)
            else:
                res += c
        return res

'''
739
'''
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0]*n
        st = []
        for i, t in enumerate(temperatures):
            while st and t > temperatures[st[-1]]:
                j = st.pop()
                ans[j] = i-j
            st.append(i)
        return ans
        
'''
84
'''
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        heights = [0] + heights + [0]
        res = 0
        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                tmp = stack.pop()
                res = max(res, (i - stack[-1] - 1) * heights[tmp])
            stack.append(i)
        return res
    
'''
215
'''
class Solution:
    def findKthLargest(self, nums, k):
        def quick_select(nums, k):
            # 随机选择基准数
            pivot = random.choice(nums)
            big, equal, small = [], [], []
            # 将大于、小于、等于 pivot 的元素划分至 big, small, equal 中
            for num in nums:
                if num > pivot:
                    big.append(num)
                elif num < pivot:
                    small.append(num)
                else:
                    equal.append(num)
            if k <= len(big):
                # 第 k 大元素在 big 中，递归划分
                return quick_select(big, k)
            if len(nums) - len(small) < k:
                # 第 k 大元素在 small 中，递归划分
                return quick_select(small, k - len(nums) + len(small))
            # 第 k 大元素在 equal 中，直接返回 pivot
            return pivot
        
        return quick_select(nums, k)

'''
347
'''
import heapq
from collections import defaultdict

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count_dict = defaultdict(int)
        heap = []
        res = []
        for num in nums:
            count_dict[num] += 1
        for key, val in count_dict.items():
            heapq.heappush(heap, (-val, key))
        for _ in range(k):
            res.append(heapq.heappop(heap)[1])
        return res

'''
295
'''
import heapq
class MedianFinder:

    def __init__(self):
        self.max_heap = []
        self.min_heap = []

    def addNum(self, num: int) -> None:
        heapq.heappush(self.max_heap, -num)
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        if len(self.min_heap) > len(self.max_heap) + 1:
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self) -> float:
        if len(self.min_heap) > len(self.max_heap):
            return self.min_heap[0]
        else:
            return (self.min_heap[0] - self.max_heap[0]) / 2 

'''
121
'''
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        cost, profit = float('+inf'), 0
        for price in prices:
            cost = min(cost, price)
            profit = max(profit, price - cost)
        return profit
    
'''
55
'''
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_idx = 0
        for idx, jump in enumerate(nums):
            if max_idx >= idx and idx+jump > max_idx:
                max_idx = idx+jump
        return max_idx >= idx
    
'''
45
'''
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1:return 0
        jump_cnt, cur_end, farthest = 0, 0, 0
        for idx in range(len(nums) - 1):
            farthest = max(farthest, idx+nums[idx])
            if idx == cur_end:
                jump_cnt += 1
                cur_end = farthest
        return jump_cnt
            
'''
763
'''
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = {}
        for i, x in enumerate(s):
            last[x] = i
        
        res = []
        far = -1
        start = 0
        for i in range(len(s)):
            far = max(far, last[s[i]])
            if i == far:  
                res.append(far - start + 1)
                start = far + 1
        return res
    
'''
70
'''
class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 1: return
        dp = [0] * (n+1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]
    
'''
118
'''
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]*n for n in range(1, numRows+1)]
        # n = sum([num for num in range(numRows, -1, -1)])
        for r_idx,row in enumerate(res[2:]):
            for idx in range(1, len(row)-1):
                res[r_idx+2][idx] = res[r_idx+1][idx-1] + res[r_idx+1][idx]
        return res

'''
198
'''
class Solution:
    def rob(self, nums: List[int]) -> int:
        '''
        dp[i] = f(n)
        dp[n+1] = max(dp[n], dp[n-1]+num)
        '''
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        dp = [0]*n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        return dp[-1]
    
'''
279
'''
class Solution:
    def numSquares(self, n: int) -> int:
        '''
        dp[i] = min{dp[i-j^2]+1}
        '''
        dp = [0]*(n+1)
        for i in range(1, n + 1):
            if i ** 0.5 % 1 == 0:
                dp[i] = 1
            else:
                dp[i] = 1 + min([dp[i-j*j] for j in range(1,int(i**0.5)+1)])
        return dp[-1]

'''
322
'''
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        '''
        dp[i] = min(dp[i], dp[i-coin]+1)
        '''
        dp = [float('inf')] * (amount+1)
        dp[0] = 0
        for i in range(1, amount+1):
            for coin in coins:
                if i-coin >= 0:
                    dp[i] = min(dp[i], dp[i-coin]+1)
        return dp[-1] if dp[-1] != float('inf') else -1
    
'''
139
'''
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordSet = set(wordDict)  # 转换为集合以提高查找效率
        dp = [False] * (len(s) + 1)
        dp[0] = True  # 空字符串可以被分割

        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordSet:
                    dp[i] = True
                    break

        return dp[len(s)]
    
'''
300
'''
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return
        '''
        dp[i] = dp[i-1] + num
        dp[i] = max(dp[i], dp[j]+1)
        '''
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
    
'''
152
'''
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums: return 
        res = nums[0]
        pre_max = nums[0]
        pre_min = nums[0]
        for num in nums[1:]:
            cur_max = max(pre_max * num, pre_min * num, num)
            cur_min = min(pre_max * num, pre_min * num, num)
            res = max(res, cur_max)
            pre_max = cur_max
            pre_min = cur_min
        return res
    
'''
416
'''
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        '''
        d[i][j] = dp[i-1][j], j < num
        d[i][j] = dp[i-1][j] or dp[i-1][j-num], j >= num
        '''
        total_sum = sum(nums)
        if total_sum % 2 != 0:
            return False
        n = len(nums)
        target = total_sum // 2
        dp = [[False]*(target+1) for _ in range(n+1)]
        dp[0][0] = True

        for i in range(1, n+1):
            num = nums[i-1]
            for j in range(target+1):
                if j < num:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-num]
        return dp[n][target]
    
'''
32
'''
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        ret = 0
        lg = len(s)
        for i in range(lg):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    ret = max(ret, i - stack[-1])
        return ret

'''
62
'''

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        '''
        dp[i][j] = dp[i-1][j] + dp[i][j-1]
        '''
        dp = [[1]*n] + [[1]+[0] * (n-1) for _ in range(m-1)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]

'''
64
'''
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        dp = [[float('inf')] * cols for _ in range(rows)]

        # 初始化第一个元素
        dp[0][0] = grid[0][0]

        # 初始化第一行
        for c in range(1, cols):
            dp[0][c] = dp[0][c-1] + grid[0][c]

        # 初始化第一列
        for r in range(1, rows):
            dp[r][0] = dp[r-1][0] + grid[r][0]

        # 计算其余元素
        for r in range(1, rows):
            for c in range(1, cols):
                dp[r][c] = min(dp[r-1][c], dp[r][c-1]) + grid[r][c]

        return dp[-1][-1]

'''
5
'''
def longestPalindrome(s):
    n = len(s)
    if n < 2:
        return s

    dp = [[False] * n for _ in range(n)]
    start, max_length = 0, 1

    # 所有长度为1的子串都是回文串
    for i in range(n):
        dp[i][i] = True

    # 遍历所有可能的子串长度
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2 or dp[i + 1][j - 1]:
                    dp[i][j] = True
                    if length > max_length:
                        start = i
                        max_length = length

    return s[start:start + max_length]

# 示例
s = "babad"
print(longestPalindrome(s))
