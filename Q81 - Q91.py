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
