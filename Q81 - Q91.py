# Q81.) Search in Rotated Sorted Array II

# There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values).

# Before being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,4,4,5,6,6,7] might be rotated at pivot index 5 and become [4,5,6,6,7,0,1,2,4,4].

# Given the array nums after the rotation and an integer target, return true if target is in nums, or false if it is not in nums.

# You must decrease the overall operation steps as much as possible.

# Example 1:

# Input: nums = [2,5,6,0,0,1,2], target = 0
# Output: true
# Example 2:

# Input: nums = [2,5,6,0,0,1,2], target = 3
# Output: false
 
# Constraints:

# 1 <= nums.length <= 5000
# -104 <= nums[i] <= 104
# nums is guaranteed to be rotated at some pivot.
# -104 <= target <= 104

# Sol_81}

class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        start, end = 0, len(nums) - 1
        
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return True
            elif nums[start] < nums[mid]:
                if nums[start] <= target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            elif nums[mid] < nums[start]:
                if nums[mid] < target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
            else:
                start += 1
        
        return False

# Q82.) Remove Duplicates from Sorted List II

# Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well.

# Example 1:


# Input: head = [1,2,3,3,4,4,5]
# Output: [1,2,5]
# Example 2:


# Input: head = [1,1,1,2,3]
# Output: [2,3]
 
# Constraints:

# The number of nodes in the list is in the range [0, 300].
# -100 <= Node.val <= 100
# The list is guaranteed to be sorted in ascending order.

# Sol_82}

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head
        if head.val != head.next.val:
            sub_node = self.deleteDuplicates(head.next)
            head.next = sub_node
            return head
        else:
            cur = head
            nxt = head.next
            while nxt is not None and nxt.val == cur.val:
                nxt = nxt.next
            return self.deleteDuplicates(nxt)

# Q83.) Remove Duplicates from Sorted List

# Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

# Example 1:

# Input: head = [1,1,2]
# Output: [1,2]
# Example 2:

# Input: head = [1,1,2,3,3]
# Output: [1,2,3]
 
# Constraints:

# The number of nodes in the list is in the range [0, 300].
# -100 <= Node.val <= 100
# The list is guaranteed to be sorted in ascending order.

# Sol_83}

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        res = head

        while head and head.next:
            if head.val == head.next.val:
                head.next = head.next.next
            else:
                head = head.next
        
        return res

# Q84.) Largest Rectangle in Histogram

# Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

# Example 1:

# Input: heights = [2,1,5,6,2,3]
# Output: 10
# Explanation: The above is a histogram where width of each bar is 1.
# The largest rectangle is shown in the red area, which has an area = 10 units.
# Example 2:

# Input: heights = [2,4]
# Output: 4
 
# Constraints:

# 1 <= heights.length <= 105
# 0 <= heights[i] <= 104

# Sol_84}

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        max_area = 0

        for i in range(len(heights)):
            while stack[-1] != -1 and heights[i] <= heights[stack[-1]]:
                height = heights[stack.pop()]
                width = i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        while stack[-1] != -1:
            height = heights[stack.pop()]
            width = len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area

# Q85.) Maximal Rectangle

# Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

# Example 1:

# Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# Output: 6
# Explanation: The maximal rectangle is shown in the above picture.
# Example 2:

# Input: matrix = [["0"]]
# Output: 0
# Example 3:

# Input: matrix = [["1"]]
# Output: 1
 
# Constraints:

# rows == matrix.length
# cols == matrix[i].length
# 1 <= row, cols <= 200
# matrix[i][j] is '0' or '1'.

# Sol_85}

class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        r, c = len(matrix), len(matrix[0])
        h = [0] * (c + 1)  # Extra space for boundary condition
        maxArea = 0

        for i in range(r):
            st = [-1]
            for j in range(c):
                # Calculate height for current row
                h[j] = h[j] + 1 if matrix[i][j] == '1' else 0

            for j in range(c + 1):
                while st[-1] != -1 and h[st[-1]] > h[j]:
                    height = h[st.pop()]
                    width = j - st[-1] - 1
                    maxArea = max(maxArea, height * width)
                st.append(j)

        return maxArea

# Q86.) Partition list

# Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

# You should preserve the original relative order of the nodes in each of the two partitions.

# Example 1:

# Input: head = [1,4,3,2,5,2], x = 3
# Output: [1,2,2,4,3,5]
# Example 2:

# Input: head = [2,1], x = 2
# Output: [1,2]
 
# Constraints:

# The number of nodes in the list is in the range [0, 200].
# -100 <= Node.val <= 100
# -200 <= x <= 200

# Sol_86}

class Solution:
  def partition(self, head: ListNode, x: int) -> ListNode:
    beforeHead = ListNode(0)
    afterHead = ListNode(0)
    before = beforeHead
    after = afterHead

    while head:
      if head.val < x:
        before.next = head
        before = head
      else:
        after.next = head
        after = head
      head = head.next

    after.next = None
    before.next = afterHead.next

    return beforeHead.next

# Q87.) Scramble String

# We can scramble a string s to get a string t using the following algorithm:

# If the length of the string is 1, stop.
# If the length of the string is > 1, do the following:
# Split the string into two non-empty substrings at a random index, i.e., if the string is s, divide it to x and y where s = x + y.
# Randomly decide to swap the two substrings or to keep them in the same order. i.e., after this step, s may become s = x + y or s = y + x.
# Apply step 1 recursively on each of the two substrings x and y.
# Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false.

# Example 1:

# Input: s1 = "great", s2 = "rgeat"
# Output: true
# Explanation: One possible scenario applied on s1 is:
# "great" --> "gr/eat" // divide at random index.
# "gr/eat" --> "gr/eat" // random decision is not to swap the two substrings and keep them in order.
# "gr/eat" --> "g/r / e/at" // apply the same algorithm recursively on both substrings. divide at random index each of them.
# "g/r / e/at" --> "r/g / e/at" // random decision was to swap the first substring and to keep the second substring in the same order.
# "r/g / e/at" --> "r/g / e/ a/t" // again apply the algorithm recursively, divide "at" to "a/t".
# "r/g / e/ a/t" --> "r/g / e/ a/t" // random decision is to keep both substrings in the same order.
# The algorithm stops now, and the result string is "rgeat" which is s2.
# As one possible scenario led s1 to be scrambled to s2, we return true.
# Example 2:

# Input: s1 = "abcde", s2 = "caebd"
# Output: false
# Example 3:

# Input: s1 = "a", s2 = "a"
# Output: true
 
# Constraints:

# s1.length == s2.length
# 1 <= s1.length <= 30
# s1 and s2 consist of lowercase English letters.

# Sol_87}

class Solution:
    def isScramble(self,s1, s2):
        m ={}
        def func(s1, s2):
            if (s1, s2) in m:
                return m[(s1, s2)]
            if not sorted(s1) == sorted(s2):
                return False
            if len(s1) == 1:
                return True
            
            for i in range(1, len(s1)):
                if func(s1[:i], s2[-i:]) and func(s1[i:], s2[:-i]) or func(s1[:i], s2[:i]) and func(s1[i:], s2[i:]):
                    m[(s1, s2)] = True
                    return True
            m[(s1, s2)] = False
            return False
        return func(s1, s2)

# Q88.) Merge Sorted Array

# You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

# Merge nums1 and nums2 into a single array sorted in non-decreasing order.

# The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

# Example 1:

# Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
# Output: [1,2,2,3,5,6]
# Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
# The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
# Example 2:

# Input: nums1 = [1], m = 1, nums2 = [], n = 0
# Output: [1]
# Explanation: The arrays we are merging are [1] and [].
# The result of the merge is [1].
# Example 3:

# Input: nums1 = [0], m = 0, nums2 = [1], n = 1
# Output: [1]
# Explanation: The arrays we are merging are [] and [1].
# The result of the merge is [1].
# Note that because m = 0, there are no elements in nums1. The 0 is only there to ensure the merge result can fit in nums1.
 
# Constraints:

# nums1.length == m + n
# nums2.length == n
# 0 <= m, n <= 200
# 1 <= m + n <= 200
# -109 <= nums1[i], nums2[j] <= 109
 
# Follow up: Can you come up with an algorithm that runs in O(m + n) time?

# Sol_88}

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        last = m + n - 1

        i = m - 1
        j = n - 1

        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[last] = nums1[i]
                i -= 1
            else:
                nums1[last] = nums2[j]
                j -= 1
            last -= 1

        while j >= 0:
            nums1[last] = nums2[j]
            j -= 1
            last -= 1

# Q89.) Gray Code

# An n-bit gray code sequence is a sequence of 2n integers where:

# Every integer is in the inclusive range [0, 2n - 1],
# The first integer is 0,
# An integer appears no more than once in the sequence,
# The binary representation of every pair of adjacent integers differs by exactly one bit, and
# The binary representation of the first and last integers differs by exactly one bit.
# Given an integer n, return any valid n-bit gray code sequence.

# Example 1:

# Input: n = 2
# Output: [0,1,3,2]
# Explanation:
# The binary representation of [0,1,3,2] is [00,01,11,10].
# - 00 and 01 differ by one bit
# - 01 and 11 differ by one bit
# - 11 and 10 differ by one bit
# - 10 and 00 differ by one bit
# [0,2,3,1] is also a valid gray code sequence, whose binary representation is [00,10,11,01].
# - 00 and 10 differ by one bit
# - 10 and 11 differ by one bit
# - 11 and 01 differ by one bit
# - 01 and 00 differ by one bit
# Example 2:

# Input: n = 1
# Output: [0,1]
 
# Constraints:

# 1 <= n <= 16

# Sol_89}

class Solution:
    def grayCode(self, n: int) -> List[int]:
        result = [0]
        for i in range(n):
            temp = reversed(result)
            for x in temp:
                result += [x + (1 << i)]
        return result

# Q90.) Subsets II

# Given an integer array nums that may contain duplicates, return all possible 
# subsets
#  (the power set).

# The solution set must not contain duplicate subsets. Return the solution in any order.

# Example 1:

# Input: nums = [1,2,2]
# Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
# Example 2:

# Input: nums = [0]
# Output: [[],[0]]
 
# Constraints:

# 1 <= nums.length <= 10
# -10 <= nums[i] <= 10

# Sol_90}

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []

        nums.sort()

        def backtrack(i, subset):
            if i >= len(nums):
                res.append(subset[:])
                return

            subset.append(nums[i])
            backtrack(i + 1, subset)
            subset.pop()

            while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                i += 1
            backtrack(i + 1, subset)


        backtrack(0, [])    

        return res

# Q91.) Decode Ways

# You have intercepted a secret message encoded as a string of numbers. The message is decoded via the following mapping:

# "1" -> 'A'

# "2" -> 'B'

# ...

# "25" -> 'Y'

# "26" -> 'Z'

# However, while decoding the message, you realize that there are many different ways you can decode the message because some codes are contained in other codes ("2" and "5" vs "25").

# For example, "11106" can be decoded into:

# "AAJF" with the grouping (1, 1, 10, 6)
# "KJF" with the grouping (11, 10, 6)
# The grouping (1, 11, 06) is invalid because "06" is not a valid code (only "6" is valid).
# Note: there may be strings that are impossible to decode.

# Given a string s containing only digits, return the number of ways to decode it. If the entire string cannot be decoded in any valid way, return 0.

# The test cases are generated so that the answer fits in a 32-bit integer.

# Example 1:
# Input: s = "12"

# Output: 2

# Explanation:
# "12" could be decoded as "AB" (1 2) or "L" (12).

# Example 2:
# Input: s = "226"

# Output: 3

# Explanation:
# "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).

# Example 3:
# Input: s = "06"

# Output: 0

# Explanation:
# "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06"). In this case, the string is not a valid encoding, so return 0.

# Constraints:

# 1 <= s.length <= 100
# s contains only digits and may contain leading zero(s).

# Sol_91}

class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        prev = 1
        prev_prev = 0
        for i in range(n - 1, -1, -1):
            if s[i] == "0":
                curr = 0
            else:
                curr = prev
                if (i + 1 < n) and (s[i] == "1" or (s[i] == "2" and s[i + 1] in "0123456")):
                    curr += prev_prev
            prev_prev = prev
            prev = curr
        return prev


# Q 1415.) The k-th Lexicographical String of All Happy Strings of Length n
# A happy string is a string that:

# consists only of letters of the set ['a', 'b', 'c'].
# s[i] != s[i + 1] for all values of i from 1 to s.length - 1 (string is 1-indexed).
# For example, strings "abc", "ac", "b" and "abcbabcbcb" are all happy strings and strings "aa", "baa" and "ababbc" are not happy strings.

# Given two integers n and k, consider a list of all happy strings of length n sorted in lexicographical order.
# Return the kth string of this list or return an empty string if there are less than k happy strings of length n.

# Example 1:

# Input: n = 1, k = 3
# Output: "c"
# Explanation: The list ["a", "b", "c"] contains all happy strings of length 1. The third string is "c".
# Example 2:

# Input: n = 1, k = 4
# Output: ""
# Explanation: There are only 3 happy strings of length 1.
# Example 3:

# Input: n = 3, k = 9
# Output: "cab"
# Explanation: There are 12 different happy string of length 3 ["aba", "abc", "aca", "acb", "bab", "bac", "bca", "bcb", "cab", "cac", "cba", "cbc"]. You will find the 9th string = "cab"
 
# Constraints:

# 1 <= n <= 10
# 1 <= k <= 100

# Sol_1415 }

class Solution:
    def getHappyString(self, n: int, k: int) -> str:
        totalHappy = 3 * (2**(n-1))
        if k > totalHappy:
            return ""
        
        output = ""
        choices = "abc"
        low, high = 1, totalHappy
        for i in range(n):
            partitionSize = (high - low + 1) // len(choices)
            cur = low
            for c in choices:
                if k in range(cur, cur + partitionSize):
                    output += c
                    low = cur
                    high = cur + partitionSize - 1
                    choices = "abc".replace(c, "")
                    break
                cur += partitionSize
        return output

# ============================== OR =====================================

class Solution:
    def getHappyString(self, n: int, k: int) -> str:
        totalHappy = 3 * (2**(n-1))
        if k > totalHappy:
            return ""
        
        output = ""
        choices = "abc"
        low, high = 1, totalHappy
        for i in range(n):
            partitionSize = (high - low + 1) // len(choices)
            cur = low
            for c in choices:
                if k in range(cur, cur + partitionSize):
                    output += c
                    low = cur
                    high = cur + partitionSize - 1
                    choices = "abc".replace(c, "")
                    break
                cur += partitionSize
        return output


# Q 1980.) Find Unique Binary String
# Given an array of strings nums containing n unique binary strings each of length n, return a binary string of length n that does not appear in nums. If there are multiple answers, you may return any of them.

# Example 1:

# Input: nums = ["01","10"]
# Output: "11"
# Explanation: "11" does not appear in nums. "00" would also be correct.
# Example 2:

# Input: nums = ["00","01"]
# Output: "11"
# Explanation: "11" does not appear in nums. "10" would also be correct.
# Example 3:

# Input: nums = ["111","011","001"]
# Output: "101"
# Explanation: "101" does not appear in nums. "000", "010", "100", and "110" would also be correct.
 
# Constraints:

# n == nums.length
# 1 <= n <= 16
# nums[i].length == n
# nums[i] is either '0' or '1'.
# All the strings of nums are unique.

# Sol_1980 }

class Solution(object):
    def findDifferentBinaryString(self, nums):
        res=[]
        for i in range(0, len(nums)):
            if nums[i][i]=='0':
                res.append('1')
            else:
                res.append('0')

        return "".join(res)

# ====================================================================================================

class FindElements:
    def __init__(self, root):
        self.values = set()
        self.recover_tree(root, 0)

    def recover_tree(self, node, value):
        if not node:
            return
        self.values.add(value)
        node.val = value
        self.recover_tree(node.left, 2 * value + 1)
        self.recover_tree(node.right, 2 * value + 2)

    def find(self, target):
        return target in self.values


# ================================================================================================

class Solution:
    def recoverFromPreorder(self, traversal: str) -> Optional[TreeNode]:
        # make tuples of (depth, value) for each node in tree. reverse to pop starting from root
        nodes = [(len(node[1]), int(node[2])) for node in re.findall("((-*)(\d+))", traversal)][::-1]


        def makeTree(depth): 
            # tree build when nodes empty. if expected depth != current depth then reached leaf
            if not nodes or depth != nodes[-1][0]: return None 

            # preorder traversal = root - left - right
            node = TreeNode(nodes.pop()[1]) # pop node and get value
            node.left = makeTree(depth + 1) # fill in children at depth + 1. 
            node.right = makeTree(depth + 1)

            return node

        return makeTree(0) # start building tree, returns root


# ===============================================================================================


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not postorder:
            return None
        val = postorder.pop()
        node = TreeNode(val)
        if not postorder:
            return node

        i = postorder.index(preorder[1])
        node.left = self.constructFromPrePost(preorder[1:i+2], postorder[:i+1])
        node.right = self.constructFromPrePost(preorder[i+2:], postorder[i+1:])
        return node

# ====================================================================================================

from collections import defaultdict, deque
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.kids = [] 
        self.parent = None

class Solution:
    def mostProfitablePath(self, edges: List[List[int]], bob: int, amount: List[int]) -> int:
        graph = defaultdict(list)   	
        for src, dst in edges:
            graph[src].append(dst)
            graph[dst].append(src)
        
        root = TreeNode(0)
        queue = deque([root])  
        visited = {0} 

        while queue:
            a_node = queue.popleft() 
            if a_node.key == bob:
                b_node = a_node
            for kid in graph[a_node.key]:
                if kid not in visited:
                    visited.add(kid)
                    kid_node = TreeNode(kid)
                    kid_node.parent = a_node
                    a_node.kids.append(kid_node)
                    queue.append(kid_node)

        max_alice_leaf_score = float('-inf') 

        dq = deque([(root,0)])
        while dq:
            for _ in range(len(dq)):
                a_node, score = dq.popleft() 
                a_key = a_node.key 
                a_amount = amount[a_key]

                if b_node: 
                    b_key = b_node.key 
                    if a_key == b_key:
                        a_amount /= 2

                new_score = score + a_amount
                amount[a_key] = 0

                for kid in a_node.kids:
                    dq.append((kid, new_score))
                
                if not a_node.kids:
                    max_alice_leaf_score = max(max_alice_leaf_score, new_score)
           
            if b_node: 
                b_key = b_node.key 
                amount[b_key] = 0
                b_node = b_node.parent 

        
        return int(max_alice_leaf_score)			


# ==========================================================================================

class Solution:
    def numOfSubarrays(self, arr):
        MOD = 1000000007
        odd = 0
        even = 1
        result = 0
        sum_ = 0

        for num in arr:
            sum_ += num
            if sum_ % 2 == 1:
                result = (result + even) % MOD
                odd += 1
            else:
                result = (result + odd) % MOD
                even += 1

        return result

# =======================================================================================

import math
class Solution:
    def numOfSubarrays(self, arr: List[int]) -> int:
        res = odd = even = 0
        for i in arr:
            even += 1
            if i & 1:
                odd, even = even, odd
            res += odd
        return res % (10**9 + 7)


# =======================================================================================

class Solution:
    def maxAbsoluteSum(self, nums):
        min_prefix_sum = 0
        max_prefix_sum = 0
        prefix_sum = 0

        for num in nums:
            prefix_sum += num

            min_prefix_sum = min(min_prefix_sum, prefix_sum)
            max_prefix_sum = max(max_prefix_sum, prefix_sum)

        return max_prefix_sum - min_prefix_sum

# =========================================================================================

class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
     
        dp = [[0]*(m+1) for _ in range(n+1)]
        
        for i in range(n-1, -1, -1):
            for j in range(m-1, -1, -1):
                if str1[i] == str2[j]:
                    dp[i][j] = dp[i+1][j+1]+1
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j+1])
        
        i = 0
        j = 0
        ans = ""

        while i < n and j < m:
            if str1[i] == str2[j]:
                ans += str1[i]
                i += 1
                j += 1
            elif dp[i+1][j] > dp[i][j+1]:
                ans += str1[i]
                i+=1
            else:
                ans += str2[j]
                j+=1
        
        while i < n:
            ans += str1[i]
            i += 1
        
        while j < m:
            ans += str2[j]
            j += 1

        return ans

# =========================================================================================


class Solution:
    def applyOperations(self, nums: List[int]) -> List[int]:
        n = len(nums)

        for i in range(n - 1):
            if nums[i] == nums[i + 1]:
                nums[i] *= 2
                nums[i + 1] = 0

        non_zero_idx = 0

        for i in range(n):
            if nums[i] != 0:
                nums[non_zero_idx] = nums[i]
                non_zero_idx += 1

        for i in range(non_zero_idx, n):
            nums[i] = 0
        
        return nums
# ======================================================================================

class Solution:
    def minimumRecolors(self, blocks: str, k: int) -> int:
        black_count = 0
        ans = float("inf")
        for i in range(len(blocks)):
            if i - k >= 0 and blocks[i - k] == 'B': 
                black_count -= 1
            if blocks[i] == 'B':
                black_count += 1            
            ans = min(ans, k - black_count)
        
        return ans

# =======================================================================================

class Solution:
    def minimumRecolors(self, b: str, k: int) -> int:
        c=0
        for i in range(k):
            if b[i]=='W':
                c+=1
        a,l=c,0
        for i in range(k,len(b)):
            if b[l]=='W':
                c-=1
            l+=1
            if b[i]=='W':
                c+=1
            a=min(a,c)
        return a

# ========================================================================================

class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set("aeiou")
        n = len(word)
        result = 0
         # Precompute next_consonant: for every index, store the next index where a consonant occurs.
        next_consonant = [n] * n
        next_cons_index = n
        for i in range(n - 1, -1, -1):
            next_consonant[i] = next_cons_index
            if word[i] not in vowels:
                next_cons_index = i

        vowel_count = {}
        cons_count = 0
        left = 0  # left pointer for the sliding window

        for right in range(n):
            ch = word[right]
            if ch in vowels:
                vowel_count[ch] = vowel_count.get(ch, 0) + 1
            else:
                cons_count += 1

            # If too many consonants, shrink from the left.
            while cons_count > k and left <= right:
                left_ch = word[left]
                if left_ch in vowels:
                    vowel_count[left_ch] -= 1
                    if vowel_count[left_ch] == 0:
                        del vowel_count[left_ch]
                else:
                    cons_count -= 1
                left += 1

            # When the current window has exactly k consonants and contains all vowels,
            # count all valid substrings formed by extending the window with following vowels.
            while left <= right and cons_count == k and len(vowel_count) == 5:
                # All substrings from current valid window ending at 'right' and extending till
                # right before the next consonant are valid.
                result += next_consonant[right] - right

                # Move left pointer to try for a new valid window.
                left_ch = word[left]
                if left_ch in vowels:
                    vowel_count[left_ch] -= 1
                    if vowel_count[left_ch] == 0:
                        del vowel_count[left_ch]
                else:
                    cons_count -= 1
                left += 1

        return result

# =====================================================================================================

class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        def f(k: int) -> int:
            cnt = Counter()
            ans = l = x = 0
            for c in word:
                if c in "aeiou":
                    cnt[c] += 1
                else:
                    x += 1
                while x >= k and len(cnt) == 5:
                    d = word[l]
                    if d in "aeiou":
                        cnt[d] -= 1
                        if cnt[d] == 0:
                            cnt.pop(d)
                    else:
                        x -= 1
                    l += 1
                ans += l
            return ans

        return f(k) - f(k + 1)
# =========================================================================================================

class Solution(object):
    def minZeroArray(self, nums, queries):
        n = len(nums)
        sum_value = 0
        query_count = 0
        diff_array = [0] * (n + 1)
        for i in range(n):
            while sum_value + diff_array[i] < nums[i]:
                query_count += 1
                if query_count > len(queries):
                    return -1
                left, right, value = queries[query_count - 1]
                if right >= i:
                    diff_array[max(left, i)] += value
                    if right + 1 < len(diff_array):
                        diff_array[right + 1] -= value
            sum_value += diff_array[i]
        return query_count
# =====================================================================================================================
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last_occurrence = {}
        
        for i, char in enumerate(s):
            last_occurrence[char] = i
        
        result = []
        start = 0
        end = 0
        
        for i, char in enumerate(s):
            end = max(end, last_occurrence[char])
            
            if i == end:
                result.append(end - start + 1)
                start = i + 1
        
        return result
# ==================================================================================================================
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        hmap = {}
        n = len(s)
        result = []

        for i, ele in enumerate(s):
            hmap[ele] = i

        cur = 0
        ans = -1
        for i, ele in enumerate(s):
            cur += 1
            if hmap[ele] > ans:
                ans = hmap[ele]

            if i == ans:
                result.append(cur)
                cur = 0

        return result
# =====================================================================================================================
class Solution(object):
    def putMarbles(self, weights, k):
        n = len(weights) - 1
        weights = [weights[i] + weights[i + 1] for i in range(n)]
        weights.sort()
        res = 0
        for i in range(k - 1):
            res += weights[-1 - i] - weights[i]
        return res
# ================================================================================================================
class Solution:
    def putMarbles(self, weights: List[int], k: int) -> int:
        if len(weights) == k or k == 1:
            return 0

        wt_sum = [weights[i]+weights[i+1] for i in range(len(weights)-1)]
        wt_sum.sort()

        return sum(wt_sum[-(k-1):]) - sum(wt_sum[:k-1])
# ================================================================================================================
class Solution(object):
    def lcaDeepestLeaves(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """
        def dfs(node, depth):
            if not node:
                # If node is None, return (None, current depth)
                return (None, depth)
            
            # Recursively traverse left and right children
            left_lca, left_depth = dfs(node.left, depth + 1)
            right_lca, right_depth = dfs(node.right, depth + 1)

            if left_depth > right_depth:
                # Left subtree is deeper → propagate its LCA and depth upward
                return (left_lca, left_depth)
            elif right_depth > left_depth:
                # Right subtree is deeper → propagate its LCA and depth upward
                return (right_lca, right_depth)
            else:
                # Both subtrees are at the same depth → current node is LCA
                return (node, left_depth)

        # Start recursive DFS from the root at depth 0
        lca_node, _ = dfs(root, 0)
        return lca_node
# =====================================================================================================
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        hashmap = {}
        queue = [root]
        leaves = []
        while queue:
            temp = []
            for i in queue:
                if i.left:
                    hashmap[i.left] = i
                    temp += [i.left]
                if i.right:
                    hashmap[i.right] = i
                    temp += [i.right]
            if temp == []:
                leaves = queue
            queue = temp
            
        n = len(leaves)
            
        if n == 1:
            return leaves[0]
        
        ans = root
            
        while True:
            flag = True
            leaves[0] = hashmap[leaves[0]]
            
            for i in range(1, n):
                leaves[i] = hashmap[leaves[i]]
                if leaves[i] != leaves[i-1]:
                    flag = False
            if flag:
                ans = leaves[0]
                break
                
        return ans
# =========================================================================================================
class Solution:
    def subsetXORSum(self, nums):
        total = 0
        for num in nums:
            total |= num  # Step 1: Compute bitwise OR of all numbers
        return total * (1 << (len(nums) - 1))  # Step 2: Multiply by 2^(n-1)
# ============================================================================================================
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        xor_list = [0]
        for i in nums:
            for j in range(len(xor_list)):
                xor_list.append(i^xor_list[j])
        return sum(xor_list)
# =========================================================================================================
class Solution(object):
    def largestDivisibleSubset(self, nums):
        nums.sort()
        dp = [1] * len(nums)
        prev = [-1] * len(nums)
        maxi = 0
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] % nums[j] == 0 and dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
                    prev[i] = j
            if dp[i] > dp[maxi]:
                maxi = i
        res = []
        i = maxi
        while i >= 0:
            res.append(nums[i])
            i = prev[i]
        return res
# =============================================================================================================
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        d = {}
        nums = sorted(nums)
        d[1] = [[nums[0]]]
        max_level = 1
        for num in nums[1:]:
            inserted = False
            for level in range(max_level, 0, -1):
                for l in d[level]:
                    if num % l[-1] == 0:
                        if level + 1 > max_level:
                            d[level + 1] = [l + [num]]
                            max_level += 1
                        else:
                            d[level + 1] += [l + [num]]
                        inserted = True
                        break
                if inserted:
                    break
                if level == 1 and l == d[level][-1]:
                    d[1] += [[num]]
        return d[max_level][0]
# ==================================================================================
class Solution:
    def canPartition(self, nums):
        total = sum(nums)
        if total % 2 != 0:
            return False
        
        target = total // 2
        possible_sums = set([0])
        
        for num in nums:
            next_sums = set()
            for s in possible_sums:
                if s + num == target:
                    return True
                next_sums.add(s + num)
                next_sums.add(s)
            possible_sums = next_sums
        
        return target in possible_sums
# ========================================================================================
class Solution(object):
    def canPartition(self, nums):
        totalSum = sum(nums)
        if totalSum % 2 != 0:
            return False
        targetSum = totalSum // 2
        dp = [False] * (targetSum + 1)
        dp[0] = True
        for num in nums:
            for currSum in range(targetSum, num - 1, -1):
                dp[currSum] = dp[currSum] or dp[currSum - num]
        return dp[targetSum]
# =======================================================================================

class Solution:
    def minimumOperations(self, nums):
        cnt = 0
        while True:
            mpp = {}
            temp = 0
            for num in nums:
                mpp[num] = mpp.get(num, 0) + 1
                if mpp[num] == 2:
                    temp += 1
            if temp == 0:
                break
            nums = nums[min(3, len(nums)):]
            cnt += 1
        return cnt
# ==================================================================================
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # c = 0
        # while len(list(set(nums)))!=len(nums):
        #     nums = nums[3:]
        #     c+=1
        # return c
        seen = set()
        for i in range(len(nums) - 1,  -1, -1):
            if nums[i] in seen:
                return i // 3 + 1
            seen.add(nums[i])
        return 0
# ==========================================================================================
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:

        """
        def check_unique(start):
            seen = set()
            for num in nums[start:]:
                if num in seen:
                    return False
                seen.add(num)
            return True

        ans = 0
        for i in range(0, len(nums), 3):
            if check_unique(i):
                return ans
            ans += 1
        return ans
        """

        seen = [False] * 128

        for i in range(len(nums) - 1, -1, -1):
            if seen[nums[i]]:
                return i // 3 + 1
            seen[nums[i]] = True

        return 0
# ========================================================================================
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count(val: int) -> int:
            chakra = str(val)  # Chakra flow string
            n = len(chakra) - len(s)  # How much room left for chakra prefix

            if n < 0:
                return 0  # Not enough space to include suffix

            # dp[i][tight] = number of valid chakra flows from index i
            dp = [[0] * 2 for _ in range(n + 1)]

            # Base case: formed entire prefix, check suffix compatibility
            dp[n][0] = 1
            dp[n][1] = int(chakra[n:] >= s)

            # Fill DP table from back to front
            for i in range(n - 1, -1, -1):
                digit = int(chakra[i])

                # Not tight → any digit from 0 to limit
                dp[i][0] = (limit + 1) * dp[i + 1][0]

                # Tight case → we must respect current digit
                if digit <= limit:
                    dp[i][1] = digit * dp[i + 1][0] + dp[i + 1][1]
                else:
                    dp[i][1] = (limit + 1) * dp[i + 1][0]

            return dp[0][1]

        return count(finish) - count(start - 1)
# ================================================================================================
class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        count = 0  # 🍜 Mission count

        for num in range(low, high + 1):
            s = str(num)  # 🔍 Naruto's string transformation no jutsu
            n = len(s)

            if n % 2 != 0:
                continue  # ☠️ Odd-digit numbers are not balanced, skip

            half = n // 2
            left = sum(int(s[i]) for i in range(half))  # ⬅️ Left chakra
            right = sum(int(s[i]) for i in range(half, n))  # ➡️ Right chakra

            if left == right:
                count += 1  # ✅ Symmetry detected, add to mission count

        return count
# =============================================================================================
class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        ans=0
        for i in range(low,high+1):
            ss=str(i)
            m=len(ss)
            if m%2==0:
                n=m//2
                x1=0
                x2=0
                for i in range(n):
                    x1+=ord(ss[i])-ord('0')
                    x2+=ord(ss[i+n])-ord('0')
                if x1==x2: ans+=1
        return ans
# ====================================================================================================
class Solution:
    MOD = 10**9 + 7

    def countGoodNumbers(self, chakra_length: int) -> int:
        even_positions = (chakra_length + 1) // 2
        odd_positions = chakra_length // 2

        even_ways = self.chakra_power(5, even_positions)
        odd_ways = self.chakra_power(4, odd_positions)

        return (even_ways * odd_ways) % self.MOD

    def chakra_power(self, base, power):
        result = 1
        base %= self.MOD

        while power > 0:
            if power % 2 == 1:
                result = (result * base) % self.MOD
            base = (base * base) % self.MOD
            power //= 2

        return result
# =====================================================================================================

class Solution:
    def countGoodNumbers(self, n: int) -> int:    

        MOD = 10 ** 9 + 7
        
        def expo(x: int, num: int) -> int:
            if num == 0:
                return 1  
            elif num % 2 == 0:
                return expo(x ** 2 % MOD, num // 2)
            return x * expo(x, num - 1) % MOD

        return 5 ** (n % 2) * expo(20, n // 2) % MOD
# =======================================================================================================
class Solution:
    def goodTriplets(self, nums1, nums2):
        n = len(nums1)
        pos2 = {nums2[i]: i for i in range(n)}

        # Map nums1[i] to its position in nums2
        mapped = [pos2[nums1[i]] for i in range(n)]

        bitLeft = BIT(n)
        bitRight = BIT(n)
        rightCount = [0] * n

        # Count right side first
        for i in range(n - 1, -1, -1):
            rightCount[i] = bitRight.query(n - 1) - bitRight.query(mapped[i])
            bitRight.update(mapped[i], 1)

        result = 0

        # Count left and calculate triplets
        for i in range(n):
            left = bitLeft.query(mapped[i] - 1)
            right = rightCount[i]
            result += left * right
            bitLeft.update(mapped[i], 1)

        return result


class BIT:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 2)

    def update(self, index, delta):
        index += 1
        while index <= self.size + 1:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        result = 0
        index += 1
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result
# ===============================================================================================
class Solution:
    # (merge) sort data and for each i, invcounts[data[i]] will be the number of 
    # j < i s.t data[i] < data[j]
    def sort_and_count_inversions(self, data : List[int], invcounts : List[int], start : int, end : int) -> List[int]:
        n = end - start
        if n <= 8:
            # bubble sort
            data_out = data[start:end]
            invcounts_out = invcounts[start:end]
            while 1 < n:
                last_i = 0
                for i in range(n - 1):
                    if data_out[i+1] < data_out[i]:
                        data_out[i], data_out[i+1] = data_out[i+1], data_out[i]
                        invcounts_out[i], invcounts_out[i+1] = invcounts_out[i+1]+1, invcounts_out[i]         
                        last_i = i + 1
                n = last_i
            return data_out, invcounts_out
        # merge sort
        m = start + n // 2
        d1, c1 = self.sort_and_count_inversions(data, invcounts, start, m)
        d2, c2 = self.sort_and_count_inversions(data, invcounts, m, end)
        data_out : List(int) = []
        invcounts_out : List(int) = []     
        i1 = 0
        i2 = 0
        while i1 < len(d1) and i2 < len(d2):
            if d1[i1] < d2[i2]:
                data_out.append(d1[i1])
                invcounts_out.append(c1[i1])
                i1 += 1
            else:
                data_out.append(d2[i2])
                invcounts_out.append(c2[i2] + len(d1) - i1)
                i2 += 1
        if i1 < len(d1):
            data_out += d1[i1:]
            invcounts_out += c1[i1:]
        if i2 < len(d2):
            data_out += d2[i2:]
            invcounts_out += c2[i2:]
        return data_out, invcounts_out

    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int: 
        n = len(nums1)
        invnums1 = [0] * n
        for i in range(n):
            invnums1[nums1[i]] = i
        nums = [invnums1[nums2[i]] for i in range(n)]
        _, invcounts = self.sort_and_count_inversions(nums, [0] * n, 0, n)
        res = 0
        for i in range(n):
            # number of j s.t. j < i and nums[j] < nums[i] 
            under = i - invcounts[nums[i]]
            # number of j s.t. i < j and nums[i] < nums[j] is equal to 
            above = n - nums[i] - 1 - invcounts[nums[i]]
            res += under * above       
        return(res)
# ==========================================================================================================
class Solution:
    h = ["1"]  
    def countAndSay(self, n: int) -> str:
        def u(n):
            if n <= len(Solution.h):  
                return Solution.h[n - 1]  
            else:
                prev = u(n - 1)  
                result = ""
                count = 1

                for i in range(1, len(prev)):
                    if prev[i] == prev[i - 1]:  
                        count += 1
                    else:
                        result += f"{count}{prev[i - 1]}"  
                        count = 1  

                result += f"{count}{prev[-1]}"
                Solution.h.append(result)  
                return result
        return u(n)
# ==============================================================================================================
class Solution(object):
    def numRabbits(self, answers):
        mpp = Counter(answers)
        total = 0
        for x in mpp:
            total += ceil(float(mpp[x]) / (x + 1)) * (x + 1)
        return int(total)
# =============================================================================================================
class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        ans = 0
        c = Counter(answers)
        for k, v in c.items():
            q, r = divmod(v, k+1)
            ans += q*(k+1) + ((k+1) if r > 0 else 0)
        return ans
# =============================================================================================================
class Solution(object):
    def numberOfArrays(self, differences, lower, upper):
        sum, maxi, mini = 0, 0, 0
        for x in differences:
            sum += x
            maxi = max(maxi, sum)
            mini = min(mini, sum)
        return max(0, upper - lower - maxi + mini + 1)
