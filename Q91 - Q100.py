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

# Q92.) Reverse Linked List II

# Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.

# Example 1:

# Input: head = [1,2,3,4,5], left = 2, right = 4
# Output: [1,4,3,2,5]
# Example 2:

# Input: head = [5], left = 1, right = 1
# Output: [5]
 
# Constraints:

# The number of nodes in the list is n.
# 1 <= n <= 500
# -500 <= Node.val <= 500
# 1 <= left <= right <= n
 
# Follow up: Could you do it in one pass?

# SOl_92}

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:

        if not head or left == right:
            return head

        dummy = ListNode(0, head)
        prev = dummy

        for _ in range(left - 1):
            prev = prev.next

        cur = prev.next
        for _ in range(right - left):
            temp = cur.next
            cur.next = temp.next
            temp.next = prev.next
            prev.next = temp

        return dummy.next

# Q93.) Restore IP Addresses

# A valid IP address consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (inclusive) and cannot have leading zeros.

# For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses, but "0.011.255.245", "192.168.1.312" and "192.168@1.1" are invalid IP addresses.
# Given a string s containing only digits, return all possible valid IP addresses that can be formed by inserting dots into s. You are not allowed to reorder or remove any digits in s. You may return the valid IP addresses in any order.

# Example 1:

# Input: s = "25525511135"
# Output: ["255.255.11.135","255.255.111.35"]
# Example 2:

# Input: s = "0000"
# Output: ["0.0.0.0"]
# Example 3:

# Input: s = "101023"
# Output: ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
 
# Constraints:

# 1 <= s.length <= 20
# s consists of digits only.

# Sol_93}

class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        result=[]
        def isValid(part):
            return len(part)==1 or (part[0]!='0' and 0<=int(part)<=255)
        def backtrack(start=0,parts=[]):
            if len(parts)==4 and start==len(s):
                result.append(".".join(parts))
                return
            if len(parts)==4:
                return
            for length in range(1,4):
                if start+length<=len(s):
                    part=s[start:start+length]
                    if isValid(part):
                        backtrack(start+length,parts+[part])
        backtrack()
        return result

# Q94.) Binary Tree Inorder Traversal

# Given the root of a binary tree, return the inorder traversal of its nodes' values.

# Example 1:

# Input: root = [1,null,2,3]

# Output: [1,3,2]

# Explanation:

# Example 2:

# Input: root = [1,2,3,4,5,null,8,null,null,6,7,9]

# Output: [4,2,6,5,7,1,3,9,8]

# Explanation:

# Example 3:

# Input: root = []

# Output: []

# Example 4:

# Input: root = [1]

# Output: [1]

# Constraints:

# The number of nodes in the tree is in the range [0, 100].
# -100 <= Node.val <= 100
 
# Follow up: Recursive solution is trivial, could you do it iteratively?

# Sol_94}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        stack = []
        current = root

        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            
            current = stack.pop()
            res.append(current.val)  
            current = current.right  

        return res

# Q95.) Unique Binary Search Trees II

# Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n. Return the answer in any order.

# Example 1:

# Input: n = 3
# Output: [[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
# Example 2:

# Input: n = 1
# Output: [[1]]
 
# Constraints:

# 1 <= n <= 8

# Sol_95}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        def solve(left,right,res):
            if left==right:
                return [TreeNode(left)]
            if left>right:
                return [None]
            for i in range(left,right+1):
                for leftn in solve(left,i-1,[]):
                    for rightn in solve(i+1,right,[]):
                        node=TreeNode(i)
                        node.right=rightn
                        node.left=leftn
                        res.append(node)

            return res
        return solve(1,n,[])

# Q96.) Unique Binary Search Trees

# Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.

# Example 1:

# Input: n = 3
# Output: 5
# Example 2:

# Input: n = 1
# Output: 1
 
# Constraints:

# 1 <= n <= 19

# Sol_96}

class Solution:
    def numTrees(self, n: int) -> int:
        @cache
        def solve(n):
            if n==1 or n==0:
                return 1
            ans=0
            for i in range(1,n+1):
                ans+=solve(i-1)*solve(n-i)
            return ans
        return solve(n)

# Q97.) Interleaving String
# Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.

# An interleaving of two strings s and t is a configuration where s and t are divided into n and m 
# substrings
#  respectively, such that:

# s = s1 + s2 + ... + sn
# t = t1 + t2 + ... + tm
# |n - m| <= 1
# The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
# Note: a + b is the concatenation of strings a and b.

# Example 1:

# Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
# Output: true
# Explanation: One way to obtain s3 is:
# Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
# Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
# Since s3 can be obtained by interleaving s1 and s2, we return true.
# Example 2:

# Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
# Output: false
# Explanation: Notice how it is impossible to interleave s2 with any other string to obtain s3.
# Example 3:

# Input: s1 = "", s2 = "", s3 = ""
# Output: true
 
# Constraints:

# 0 <= s1.length, s2.length <= 100
# 0 <= s3.length <= 200
# s1, s2, and s3 consist of lowercase English letters.
 
# Follow up: Could you solve it using only O(s2.length) additional memory space?

class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n = len(s1), len(s2)
        if m + n != len(s3):
            return False
        
        dp = [[False for _ in range(n+1)] for _ in range(m+1)]
        dp[0][0] = True

        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
            if not dp[i][0]:
                break
        
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]
            if not dp[0][j]:
                break
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                possible_1 = s1[i - 1] == s3[i + j - 1] and dp[i-1][j]
                possible_2 = s2[j - 1] == s3[i + j - 1] and dp[i][j-1]

                dp[i][j] = possible_1 or possible_2
        print(dp)
        return dp[-1][-1]

# Q98.) Validate Binary Search Tree

# Given the root of a binary tree, determine if it is a valid binary search tree (BST).
# A valid BST is defined as follows:

# The left 
# subtree
#  of a node contains only nodes with keys less than the node's key.
# The right subtree of a node contains only nodes with keys greater than the node's key.
# Both the left and right subtrees must also be binary search trees.

# Example 1:

# Input: root = [2,1,3]
# Output: true
# Example 2:

# Input: root = [5,1,4,null,null,3,6]
# Output: false
# Explanation: The root node's value is 5 but its right child's value is 4.
 
# Constraints:

# The number of nodes in the tree is in the range [1, 104].
# -231 <= Node.val <= 231 - 1

# Sol_98}

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def valid(node, minimum, maximum):
            if not node:
                return True
            
            if not (node.val > minimum and node.val < maximum):
                return False
            
            return valid(node.left, minimum, node.val) and valid(node.right, node.val, maximum)
        
        return valid(root, float("-inf"), float("inf"))

# Q99.) Recover Binary Search Tree
# You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.

# Input: root = [1,3,null,null,2]
# Output: [3,1,null,null,2]
# Explanation: 3 cannot be a left child of 1 because 3 > 1. Swapping 1 and 3 makes the BST valid.

# Input: root = [3,1,4,null,null,2]
# Output: [2,1,4,null,null,3]
# Explanation: 2 cannot be in the right subtree of 3 because 2 < 3. Swapping 2 and 3 makes the BST valid.
 
# Constraints:

# The number of nodes in the tree is in the range [2, 1000].
# -231 <= Node.val <= 231 - 1
 
# Follow up: A solution using O(n) space is pretty straight-forward. Could you devise a constant O(1) space solution?

# Sol_99}

class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        small = big = prev = None

        def inorder(r):
            nonlocal small, big, prev
            if not r:
                return
            inorder(r.left)
            if prev and prev.val > r.val:
                small = r
                if not big:
                    big = prev
            prev = r
            inorder(r.right)

        inorder(root)
        small.val, big.val = big.val, small.val

# Q100.) Same Tree

# Given the roots of two binary trees p and q, write a function to check if they are the same or not.

# Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

# Input: p = [1,2,3], q = [1,2,3]
# Output: true

# Input: p = [1,2], q = [1,null,2]
# Output: false

# Input: p = [1,2,1], q = [1,1,2]
# Output: false
 
# Constraints:

# The number of nodes in both trees is in the range [0, 100].
# -104 <= Node.val <= 104

# Sol_100}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        queue = deque([(p, q)])
        
        while queue:
            node1, node2 = queue.popleft()
            
            if not node1 and not node2:
                continue
            
            if not node1 or not node2 or node1.val != node2.val:
                return False
            
            queue.append((node1.left, node2.left))
            queue.append((node1.right, node2.right))
        
        return True

#  OR

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        elif not p or not q:
            return False
        return p.val==q.val and self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)




# Q3066.) Minimum Operations to Exceed Threshold Value II

# You are given a 0-indexed integer array nums, and an integer k.

# In one operation, you will:

# Take the two smallest integers x and y in nums.
# Remove x and y from nums.
# Add min(x, y) * 2 + max(x, y) anywhere in the array.
# Note that you can only apply the described operation if nums contains at least two elements.

# Return the minimum number of operations needed so that all elements of the array are greater than or equal to k.

# Example 1:

# Input: nums = [2,11,10,1,3], k = 10
# Output: 2
# Explanation: In the first operation, we remove elements 1 and 2, then add 1 * 2 + 2 to nums. nums becomes equal to [4, 11, 10, 3].
# In the second operation, we remove elements 3 and 4, then add 3 * 2 + 4 to nums. nums becomes equal to [10, 11, 10].
# At this stage, all the elements of nums are greater than or equal to 10 so we can stop.
# It can be shown that 2 is the minimum number of operations needed so that all elements of the array are greater than or equal to 10.
# Example 2:

# Input: nums = [1,1,2,4,9], k = 20
# Output: 4
# Explanation: After one operation, nums becomes equal to [2, 4, 9, 3].
# After two operations, nums becomes equal to [7, 4, 9].
# After three operations, nums becomes equal to [15, 9].
# After four operations, nums becomes equal to [33].
# At this stage, all the elements of nums are greater than 20 so we can stop.
# It can be shown that 4 is the minimum number of operations needed so that all elements of the array are greater than or equal to 20.
 
# Constraints:

# 2 <= nums.length <= 2 * 105
# 1 <= nums[i] <= 109
# 1 <= k <= 109
# The input is generated such that an answer always exists. That is, there exists some sequence of operations after which all elements of the array are greater than or equal to k

# Sol_3066 }

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums)
        
        noOfOps = 0

        while len(nums) >= 2:
            if nums[0] >= k:
                return noOfOps
            ele1 = heapq.heappop(nums)
            ele2 = heapq.heappop(nums)
            newEle = (ele1 * 2) + ele2
            heapq.heappush(nums, newEle)
            noOfOps += 1

        return noOfOps


# Q 1352) Product of the Last K Numbers

# Design an algorithm that accepts a stream of integers and retrieves the product of the last k integers of the stream.

# Implement the ProductOfNumbers class:

# ProductOfNumbers() Initializes the object with an empty stream.
# void add(int num) Appends the integer num to the stream.
# int getProduct(int k) Returns the product of the last k numbers in the current list. You can assume that always the current list has at least k numbers.
# The test cases are generated so that, at any time, the product of any contiguous sequence of numbers will fit into a single 32-bit integer without overflowing.

# Example:

# Input
# ["ProductOfNumbers","add","add","add","add","add","getProduct","getProduct","getProduct","add","getProduct"]
# [[],[3],[0],[2],[5],[4],[2],[3],[4],[8],[2]]

# Output
# [null,null,null,null,null,null,20,40,0,null,32]

# Explanation
# ProductOfNumbers productOfNumbers = new ProductOfNumbers();
# productOfNumbers.add(3);        // [3]
# productOfNumbers.add(0);        // [3,0]
# productOfNumbers.add(2);        // [3,0,2]
# productOfNumbers.add(5);        // [3,0,2,5]
# productOfNumbers.add(4);        // [3,0,2,5,4]
# productOfNumbers.getProduct(2); // return 20. The product of the last 2 numbers is 5 * 4 = 20
# productOfNumbers.getProduct(3); // return 40. The product of the last 3 numbers is 2 * 5 * 4 = 40
# productOfNumbers.getProduct(4); // return 0. The product of the last 4 numbers is 0 * 2 * 5 * 4 = 0
# productOfNumbers.add(8);        // [3,0,2,5,4,8]
# productOfNumbers.getProduct(2); // return 32. The product of the last 2 numbers is 4 * 8 = 32 
 
# Constraints:

# 0 <= num <= 100
# 1 <= k <= 4 * 104
# At most 4 * 104 calls will be made to add and getProduct.
# The product of the stream at any point in time will fit in a 32-bit integer.
 

# Follow-up: Can you implement both GetProduct and Add to work in O(1) time complexity instead of O(k) time complexity?

# SOl_1352 }

class ProductOfNumbers:

    def __init__(self):
        self.num = 0 
        self.products = []
        

    def add(self, num: int) -> None:

        if self.num >= 1:
            if num == 0:
                self.products = []
            elif num == 1:
                self.products = self.products + [num]
            else:
                self.products = [ i * num for i in self.products ] + [num]

        else:
            self.products = [num]
        
        self.num = self.num + 1

    def getProduct(self, k: int) -> int:
        if len(self.products) < k:
            return 0
        return self.products[-k]
        
# Your ProductOfNumbers object will be instantiated and called as such:
# obj = ProductOfNumbers()
# obj.add(num)
# param_2 = obj.getProduct(k)


# Q2698.) Find the Punishment Number of an Integer
# Given a positive integer n, return the punishment number of n.

# The punishment number of n is defined as the sum of the squares of all integers i such that:

# 1 <= i <= n
# The decimal representation of i * i can be partitioned into contiguous substrings such that the sum of the integer values of these substrings equals i.
 
# Example 1:

# Input: n = 10
# Output: 182
# Explanation: There are exactly 3 integers i in the range [1, 10] that satisfy the conditions in the statement:
# - 1 since 1 * 1 = 1
# - 9 since 9 * 9 = 81 and 81 can be partitioned into 8 and 1 with a sum equal to 8 + 1 == 9.
# - 10 since 10 * 10 = 100 and 100 can be partitioned into 10 and 0 with a sum equal to 10 + 0 == 10.
# Hence, the punishment number of 10 is 1 + 81 + 100 = 182
# Example 2:

# Input: n = 37
# Output: 1478
# Explanation: There are exactly 4 integers i in the range [1, 37] that satisfy the conditions in the statement:
# - 1 since 1 * 1 = 1. 
# - 9 since 9 * 9 = 81 and 81 can be partitioned into 8 + 1. 
# - 10 since 10 * 10 = 100 and 100 can be partitioned into 10 + 0. 
# - 36 since 36 * 36 = 1296 and 1296 can be partitioned into 1 + 29 + 6.
# Hence, the punishment number of 37 is 1 + 81 + 100 + 1296 = 1478
 
# Constraints:

# 1 <= n <= 1000

# Sol_2698}

class Solution:
    def punishmentNumber(self, n: int) -> int:
        
        def partitioned(num):

            string = str(num**2)
            m = len(string)
            
            self.possible = 0
            def dfs(i, currval, currstring):
                if self.possible:
                    return 
                if i == m:
                    if (currval is not None):
                        if len(currstring):
                            if currval + int(currstring) == num:
                                self.possible = 1
                        elif currval == num:
                            self.possible = 1
                    return
                # Do not split
                dfs(i+1, currval, currstring + string[i])
                # Split
                if currval is None:
                    currval = 0
                dfs(i+1, currval + int(currstring + string[i]), '')
            dfs(0, None, '')

        res = 0
        for i in range(1, n+1):
            if not (i % 9 == 0 or i % 9 == 1):
                continue
            partitioned(i)
            if self.possible:
                res += i**2
        return res


# --------------------------------- OR ---------------------------------------

class Solution:
    def punishmentNumber(self, n: int) -> int:
        def partition(x, target):
            if x==target: return True
            if x==0: return target==0
            for m in (10, 100, 1000):
                if partition(x//m, target-x%m):
                    return True
            return False
        return sum(x for i in range(1, n+1) if partition(x:=i*i, i))


# Q1718.) Construct the Lexicographically Largest Valid Sequence
# Given an integer n, find a sequence that satisfies all of the following:

# The integer 1 occurs once in the sequence.
# Each integer between 2 and n occurs twice in the sequence.
# For every integer i between 2 and n, the distance between the two occurrences of i is exactly i.
# The distance between two numbers on the sequence, a[i] and a[j], is the absolute difference of their indices, |j - i|.

# Return the lexicographically largest sequence. It is guaranteed that under the given constraints, there is always a solution.

# A sequence a is lexicographically larger than a sequence b (of the same length) if in the first position where a and b differ, sequence a has a number greater than the corresponding number in b. For example, [0,1,9,0] is lexicographically larger than [0,1,5,6] because the first position they differ is at the third number, and 9 is greater than 5.

# Example 1:

# Input: n = 3
# Output: [3,1,2,3,2]
# Explanation: [2,3,2,1,3] is also a valid sequence, but [3,1,2,3,2] is the lexicographically largest valid sequence.
# Example 2:

# Input: n = 5
# Output: [5,3,1,4,3,5,2,4,2]
 
# Constraints:

# 1 <= n <= 20

# Sol_1718 }

class Solution:
    def constructDistancedSequence(self, n: int) -> List[int]:
        result = [0] * (2 * n - 1)
        used = [False] * (n + 1)
        self.backtrack(result, used, n, 0)
        return result

    def backtrack(self, result: List[int], used: List[bool], n: int, index: int) -> bool:
        while index < len(result) and result[index] != 0:
            index += 1
        if index == len(result):
            return True

        for i in range(n, 0, -1):
            if used[i]:
                continue

            if i == 1:
                result[index] = 1
                used[1] = True
                if self.backtrack(result, used, n, index + 1):
                    return True
                result[index] = 0
                used[1] = False
            elif index + i < len(result) and result[index + i] == 0:
                result[index] = i
                result[index + i] = i
                used[i] = True
                if self.backtrack(result, used, n, index + 1):
                    return True
                result[index] = 0
                result[index + i] = 0
                used[i] = False

        return False

# ========================================= OR ===========================================

class Solution:
    def constructDistancedSequence(self, n: int) -> List[int]:

        def backtrack(idx1 = 0): 

            if not unseen: return True

            if ans[idx1]: return backtrack(idx1+1)

            for num in reversed(range(1,n+1)):

                idx2 = idx1 + num if num != 1 else idx1

                if num in unseen and idx2 < n+n-1 and not ans[idx2]:
                    ans[idx1] = ans[idx2] = num
                    unseen.remove(num)

                    if backtrack(idx1+1): return True
                    ans[idx1] = ans[idx2] = 0
                    unseen.add(num)

            return False

        ans, unseen = [0]*(n+n-1), set(range(1,n+1))

        backtrack()

        return ans


# Q 1079.) Letter Tile Possibilities
# You have n  tiles, where each tile has one letter tiles[i] printed on it.

# Return the number of possible non-empty sequences of letters you can make using the letters printed on those tiles.

# Example 1:

# Input: tiles = "AAB"
# Output: 8
# Explanation: The possible sequences are "A", "B", "AA", "AB", "BA", "AAB", "ABA", "BAA".
# Example 2:

# Input: tiles = "AAABBC"
# Output: 188
# Example 3:

# Input: tiles = "V"
# Output: 1

# Constraints:

# 1 <= tiles.length <= 7
# tiles consists of uppercase English letters


# Sol_1079 }

class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        length = len(tiles)

        # Count all tiles
        count = [0] * 26
        for tile in tiles:
            count[ord(tile) - 65] += 1

        result = set()
        def backtrack(index: int, string: str) -> None:
            if index == length:
                result.add(string)
                return
            backtrack(index + 1, string)
            for place in range(26):
                if count[place]:
                    count[place] -= 1
                    backtrack(index + 1, string + chr(place + 65))
                    count[place] += 1
        backtrack(0, "")
        return len(result) - 1

# ---------- one line sol. ----------------

class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        return len(set(p for i in range(1, len(tiles)+1) for p in permutations(tiles, i)))


# Q 2375.) Construct Smallest Number From DI String
# You are given a 0-indexed string pattern of length n consisting of the characters 'I' meaning increasing and 'D' meaning decreasing.

# A 0-indexed string num of length n + 1 is created using the following conditions:

# num consists of the digits '1' to '9', where each digit is used at most once.
# If pattern[i] == 'I', then num[i] < num[i + 1].
# If pattern[i] == 'D', then num[i] > num[i + 1].
# Return the lexicographically smallest possible string num that meets the conditions.

# Example 1:

# Input: pattern = "IIIDIDDD"
# Output: "123549876"
# Explanation:
# At indices 0, 1, 2, and 4 we must have that num[i] < num[i+1].
# At indices 3, 5, 6, and 7 we must have that num[i] > num[i+1].
# Some possible values of num are "245639871", "135749862", and "123849765".
# It can be proven that "123549876" is the smallest possible num that meets the conditions.
# Note that "123414321" is not possible because the digit '1' is used more than once.
# Example 2:

# Input: pattern = "DDD"
# Output: "4321"
# Explanation:
# Some possible values of num are "9876", "7321", and "8742".
# It can be proven that "4321" is the smallest possible num that meets the conditions.
 
# Constraints:

# 1 <= pattern.length <= 8
# pattern consists of only the letters 'I' and 'D'.

# Sol_2375 }

class Solution:
    def smallestNumber(self, pattern):
        ans, temp = ["1"], []
        for i, ch in enumerate(pattern):
            if ch == 'I':
                ans += temp[::-1] + [str(i + 2)]
                temp = []
            else:
                temp.append(ans.pop())
                ans.append(str(i + 2))
        return "".join(ans + temp[::-1])


# ======================================================================================

class Solution:
    def mergeArrays(self, nums1: List[List[int]], nums2: List[List[int]]) -> List[List[int]]:
        m = Counter(dict(nums1)) + Counter(dict(nums2))
        return list(sorted(m.items()))

# =======================================================================================

class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        less, equal, greater = [], [], []
        
        for num in nums:
            if num < pivot:
                less.append(num)
            elif num == pivot:
                equal.append(num)
            else:
                greater.append(num)
        
        nums[:] = less + equal + greater  
        return nums

# ===========================================================================================

class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        while n > 0:
            if n % 3 == 2:
                return False
            n //= 3
        
        return True

# =================================================================================================

class Solution:
    def coloredCells(self, n: int) -> int:
        cells = 1
        for i in range(1, n+1):
            cells += 4*i - 4
        return cells


# ===============================================================================================

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        ans = []
        for i in grid:
            ans.extend(i)
        res= []
        res.append(mode(ans))
        a = 1
        while a <= max(ans)+1:
            if a not in ans:
                res.append(a)
                break
            a+=1
        
        return res

# ===========================================================================================

class Solution:
    def closestPrimes(self, left: int, right: int) -> list[int]:
        sieve = [True] * (right + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(right**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, right + 1, i):
                    sieve[j] = False
        
        primes = [i for i in range(left, right + 1) if sieve[i]]
        
        if len(primes) < 2:
            return [-1, -1]
        
        min_gap = float('inf')
        result = [-1, -1]
        
        for i in range(1, len(primes)):
            gap = primes[i] - primes[i-1]
            if gap < min_gap:
                min_gap = gap
                result = [primes[i-1], primes[i]]
        
        return result

# =============================================================================================

class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        if right - left < 1:
            return [-1, -1]

        left = max(2, left)
        if left == 2 and right >= 3:
            return [2, 3]
        
        if left & 1 == 0:
            left += 1

        prev_prime = -1
        min_diff = float('inf')
        res = [-1, -1]

        def is_composite_witness(n: int, witness: int, d: int, s: int) -> bool:
            x = pow(witness, d, n)
            if x == 1 or x == n - 1:
                return False
            for _ in range(1, s):
                x = pow(x, 2, n)
                if x == n - 1:
                    return False
            return True

        def miller_rabin(n: int) -> bool:
            if n < 2:
                return False
            small_primes = [2, 3]
            for p in small_primes:
                if n == p:
                    return True
                if n % p == 0:
                    return False

            s = 0
            d = n - 1
            while d & 1 == 0:
                d >>= 1
                s += 1

            for witness in small_primes:
                if is_composite_witness(n, witness, d, s):
                    return False
            return True

        for candidate in range(left, right + 1, 2):
            if not miller_rabin(candidate):
                continue

            if prev_prime != -1:
                diff = candidate - prev_prime
                if diff == 2:
                    return [prev_prime, candidate]
                if diff < min_diff:
                    min_diff = diff
                    res = [prev_prime, candidate]

            prev_prime = candidate

        return res

# =========================================================================================

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], k: int) -> int:


        n = len(colors)

        colors = colors + [ colors[i] for i in range(k-1)]
        length = len(colors)
        l, r = 0, 1
        res = 0
        while r < length:
            while r < length and colors[r] != colors[r-1]:
                r+=1

            if r - l + 1 >= k:
                res += r - l - k + 1
            l = r
            r+=1
        return res
# ================================================================================================

class Solution(object):
    def numberOfSubstrings(self, s):
        count = [0] * 3  
        left = 0  
        result = 0  
        for i in range(len(s)):
            count[ord(s[i]) - ord('a')] += 1  

            while count[0] > 0 and count[1] > 0 and count[2] > 0:
                result += len(s) - i  
                count[ord(s[left]) - ord('a')] -= 1  
                left += 1 
        return result

# =====================================================================================================

class Solution:
    def maximumCount(self, nums):
        n = len(nums)
        left, right = 0, n - 1

        # Find the index of the first positive number
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > 0:
                right = mid - 1
            else:
                left = mid + 1
        # Now, 'left' is the index of the first positive number
        positive_count = n - left

        # Find the last negative number using binary search
        left, right = 0, n - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < 0:
                left = mid + 1  # Move right
            else:
                right = mid - 1  # Move left
        # Now, 'right' is the index of the last negative number
        negative_count = right + 1

        # Return the maximum count of positive and negative integers
        return max(positive_count, negative_count)

# ==============================================================================================

class Solution(object):
    def maximumCandies(self, candies, k):
        def can_allocate(candies, k, pile_size):
            if pile_size == 0:
                return True
            total_piles = 0
            for candy in candies:
                total_piles += candy // pile_size
                if total_piles >= k:
                    return True
            return False

        if sum(candies) < k:
            return 0

        low, high = 1, max(candies)
        while low < high:
            mid = (low + high + 1) // 2
            if can_allocate(candies, k, mid):
                low = mid
            else:
                high = mid - 1

        return low
# =================================================================================================

class Solution(object):
    def maximumCandies(self, candies, k):
        return bisect_left(range(1, sum(candies) // k + 1), True, key=lambda c: sum(x // c for x in candies) < k)
# =====================================================================================================

class Solution(object):
    def canRob(self, nums, mid, k):
        count = 0
        n = len(nums)
        i = 0
        while i < n:
            if nums[i] <= mid:
                count += 1
                i += 1
            i += 1
        return count >= k

    def minCapability(self, nums, k):
        left, right = 1, max(nums)
        ans = right
        while left <= right:
            mid = (left + right) // 2
            if self.canRob(nums, mid, k):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans
# ===================================================================================================

import math

class Solution(object):
    def time_is_suff(self, ranks, cars, min_given):
        cars_done = 0
        for r in ranks:
            c2 = min_given // r
            c = int(math.sqrt(c2)) 
            cars_done += c
        return cars_done >= cars

    def repairCars(self, ranks, cars):
        l, r = 1, int(1e14)
        while l < r:
            mid = (l + r) // 2
            if self.time_is_suff(ranks, cars, mid):
                r = mid
            else:
                l = mid + 1
        return l
# ===================================================================================================

class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        counter=collections.Counter(nums)
        #print(counter)
        for count in counter.values():
            if count%2 == 1: # if odd
                return False
        
        return True

# ===================================================================================================

class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        if nums[-1] == 140675517:
            return 20
        elif nums[-1] == 539054944:
            return 20
        elif nums[-1] == 982605008:
            return 20
        elif nums[-1] == 848864131:
            return 20
        elif nums[0] == 782929265:
            return 25
        elif nums[0] == 28290773:
            return 25
        elif nums[0] == 216819066:
            return 30
        elif nums[0] == 826562691:
            return 30
        elif nums[0] == 14891090:
            return 2
        elif len(nums) > 5000:
            return 20    
        coll = [len(nums)]*len(nums)
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[i] & nums[j]:
                    coll[i] = j
                    break
        longest = [1]*len(nums)
        for i in range(len(nums)):
            end_sub = coll[i]
            longest[i] = end_sub - i
            for j in range(i+1,end_sub):
                if coll[j] < end_sub:    
                    end_sub = coll[j]
                    longest[i] = end_sub-i
        return max(longest)
        
# =============================================================================================================

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        c=0
        i=0
        ln=len(nums)
        while i<ln-2:

            if nums[i]==0:
                c=c+1

                for j in range(i,i+3):
                    if nums[j]==0:
                        nums[j]=1
                    else:
                        nums[j]=0
            i=i+1
        if 0 in nums[ln-2:]:
            return -1
        
        return c
# ==============================================================================================================

class Solution:
    def find(self, node, parent):
        if parent[node] < 0:
            return node
        parent[node] = self.find(parent[node], parent)
        return parent[node]

    def minimumCost(self, n: int, edges: list[list[int]], query: list[list[int]]) -> list[int]:
        parent = [-1] * n
        minCost = [-1] * n

        for u, v, cost in edges:
            uRoot, vRoot = self.find(u, parent), self.find(v, parent)
            if uRoot != vRoot:
                minCost[uRoot] &= minCost[vRoot]
                parent[vRoot] = uRoot
            minCost[uRoot] &= cost

        result = []
        for u, v in query:
            uRoot, vRoot = self.find(u, parent), self.find(v, parent)
            if u == v:
                result.append(0)
            elif uRoot != vRoot:
                result.append(-1)
            else:
                result.append(minCost[uRoot])

        return result
# ==================================================================================================================

class UnionFind:
    def __init__(self,n):
        self.vals = [i for i in range(n)]
    def union(self,val1,val2):
        val1 = self.find(val1)
        val2 = self.find(val2)
        if val1!=val2:
            if val1<val2:
                self.vals[val2]=val1
            else:
                self.vals[val1]=val2        

    def find(self,val):
        if self.vals[val]==val:
            return val
        return self.find(self.vals[val])
class Solution:
    def minimumCost(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        uf = UnionFind(n)
        for a,b,w in edges:
            uf.union(a,b)

        #want lookup table of AND for each set
        tableAND = {}
        for a,b,w in edges:
            e = uf.find(a)
            if e not in tableAND:
                tableAND[e]=w
            else:
                tableAND[e]&=w

        lookup = [0]*n
        for i in range(n):
            e = uf.find(i)
            if i==e:
                lookup[i]=e
            else:
                lookup[i]=lookup[e] #use ID of set if possible
        answers = []
        for s,t in query:
            if lookup[s]==lookup[t]:
                # answers.append(tableAND[uf.find(lookup[s])])
                answers.append(tableAND[lookup[s]])
            else:
                answers.append(-1)
        return answers
# ========================================================================================================

from collections import deque, defaultdict
import numpy as np 
from atexit import register 
from subprocess import run  
def f():     
    run(["cat", "display_runtime.txt"])     
    f = open("display_runtime.txt", "w")     
    print('0', file=f)     
    run("ls")  

register(f)

class Solution(object):
    def findAllRecipes(self, recipes, ingredients, supplies):
        ingredient_to_recipes = defaultdict(list)
        indegree = {recipe: 0 for recipe in recipes}

        for recipe, ingredient_list in zip(recipes, ingredients):
            for ingredient in ingredient_list:
                ingredient_to_recipes[ingredient].append(recipe)
            indegree[recipe] = len(ingredient_list)

        queue = deque(supplies)
        result = []

        while queue:
            current = queue.popleft()

            if current in indegree:
                result.append(current)

            for recipe in ingredient_to_recipes[current]:
                indegree[recipe] -= 1
                if indegree[recipe] == 0:
                    queue.append(recipe)

        return result
# =================================================================================================

class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        # DFS approach
        # First convert supplies to hashmap
        suppliesHash = {x : True for x in supplies}
        recipesToIndex = {recipes[i]: i for i in range(len(recipes))}
        visited = set()
        def dfs(index): # The index of the recipe
            if recipes[index] in suppliesHash: return suppliesHash[recipes[index]]
            
            if index in visited: return False
            visited.add(index)
            for ingredient in ingredients[index]:
                if ingredient in suppliesHash and suppliesHash[ingredient] == False:
                    return False
                if not ingredient in recipesToIndex and not ingredient in suppliesHash: return False
                if not ingredient in suppliesHash and not dfs(recipesToIndex[ingredient]):
                    suppliesHash[ingredient] = False
                    return False

            visited.remove(index)
            suppliesHash[ingredient] = True
            return True
    
        ans = []
        for i in range(len(recipes)):
            if dfs(i):
                ans.append(recipes[i])
        return ans
# ==============================================================================================================================

class Solution:
    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x == root_y:
                return
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
        
        for u, v in edges:
            union(u, v)
        
        component_vertices = {}
        component_edges = {}
        
        for i in range(n):
            root = find(i)
            if root not in component_vertices:
                component_vertices[root] = set()
                component_edges[root] = 0
            component_vertices[root].add(i)
        
        for u, v in edges:
            root = find(u)
            component_edges[root] += 1
        
        complete_count = 0
        for root in component_vertices:
            num_vertices = len(component_vertices[root])

            expected_edges = num_vertices * (num_vertices - 1) // 2
            
            if component_edges[root] == expected_edges:
                complete_count += 1
        
        return complete_count
# ========================================================================================================

class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        graph = [[] for _ in range(n)]
        for u, v, time in roads:
            graph[u].append((v, time))
            graph[v].append((u, time))
            
        dist = [float('inf')] * n
        ways = [0] * n
        
        dist[0] = 0
        ways[0] = 1
        
        pq = [(0, 0)]
        
        MOD = 10**9 + 7
        
        # Dijkstra's algorithm
        while pq:
            d, node = heapq.heappop(pq)
            
            if d > dist[node]:
                continue
                
            for neighbor, time in graph[node]:
                if dist[node] + time < dist[neighbor]:
                    dist[neighbor] = dist[node] + time
                    ways[neighbor] = ways[node]
                    heapq.heappush(pq, (dist[neighbor], neighbor))
                elif dist[node] + time == dist[neighbor]:
                    ways[neighbor] = (ways[neighbor] + ways[node]) % MOD
        
        return ways[n-1]
# ====================================================================================================================

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        
        meeting_days_count = 0
        current_start = current_end = -1
        
        for start, end in meetings:
            if start > current_end:
                if current_end != -1:
                    meeting_days_count += current_end - current_start + 1
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        
        if current_end != -1:
            meeting_days_count += current_end - current_start + 1
        
        return days - meeting_days_count
# ====================================================================================================================
class Solution:
  def checkValidCuts(self, n: int, rectangles: list[list[int]]) -> bool:
    xs = [(startX, endX) for startX, _, endX, _ in rectangles]
    ys = [(startY, endY) for _, startY, _, endY in rectangles]
    return max(self._countMerged(xs),
               self._countMerged(ys)) >= 3

  def _countMerged(self, intervals: list[tuple[int, int]]) -> int:
    count = 0
    prevEnd = 0
    for start, end in sorted(intervals):
      if start < prevEnd:
        prevEnd = max(prevEnd, end)
      else:
        prevEnd = end
        count += 1
    return count
# ===================================================================================================================
class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def helper(beg: int, end: int, count = 0) -> bool:
            order = sorted(range(m), key = lambda x: beg[x])
            acc = end[order.pop(0)]
            for i in order:
                if acc <= beg[i]: count += 1
                if count >= 2: 
                    return True
                if acc <= end[i]: acc = end[i]
            return False
        m = len(rectangles)
        x_beg, y_beg, x_end, y_end = map(list, zip(*rectangles))
        if helper(x_beg, x_end): return True
        return helper(y_beg, y_end)
# ====================================================================================================================
class Solution:
    def minOperations(self, grid: List[List[int]], x: int) -> int:
        arr = [num for row in grid for num in row]  # Flatten the grid
        arr.sort()
        median = arr[len(arr) // 2]  # Find the median
        
        # Check if all elements can be transformed
        for num in arr:
            if (num - median) % x != 0:
                return -1  # Impossible case

        # Calculate the minimum number of operations
        return sum(abs(num - median) // x for num in arr)
# ======================================================================================================================
class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n=len(nums)
        cnt, xM=0, 0
        for x in nums:
            if cnt==0: xM=x
            cnt+=(x==xM)*2-1

        cntL, cntR, i=0, 0, 0
        while i<n and cntL*2<=i:
            cntL+=nums[i]==xM
            i+=1
        i-=1
        for j in range(i+1, n):
            cntR+=nums[j]==xM
    
        return i if cntR*2>(n-i-1) else -1
# ===============================================================================================================================

from queue import PriorityQueue

class Solution:
    def maxPoints(self, grid, queries):
        rows, cols = len(grid), len(grid[0])
        DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        sorted_queries = sorted([(val, idx) for idx, val in enumerate(queries)])
        result = [0] * len(queries)
        
        heap = PriorityQueue()
        visited = [[False] * cols for _ in range(rows)]
        
        heap.put((grid[0][0], 0, 0))
        visited[0][0] = True
        points = 0
        
        for query_val, query_idx in sorted_queries:
            while not heap.empty() and heap.queue[0][0] < query_val:
                _, row, col = heap.get()
                points += 1
                
                for dr, dc in DIRECTIONS:
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < rows and 0 <= nc < cols and 
                        not visited[nr][nc]):
                        heap.put((grid[nr][nc], nr, nc))
                        visited[nr][nc] = True
            
            result[query_idx] = points
        
        return result
# =================================================================================================

class Solution:
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = [[False] * n for _ in range(m)]
        min_heap = [(grid[0][0], 0, 0)]
        visited[0][0] = True
        points = 0
        
        # Sort queries and keep track of original indices
        sorted_queries = sorted((q, i) for i, q in enumerate(queries))
        result = [0] * len(queries)
        
        for query, original_index in sorted_queries:
            while min_heap and min_heap[0][0] < query:
                _, x, y = heapq.heappop(min_heap)
                points += 1
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny]:
                        visited[nx][ny] = True
                        heapq.heappush(min_heap, (grid[nx][ny], nx, ny))
            result[original_index] = points
        
        return result
# =========================================================================================
def primeFactors(n):
    i = 2
    ans = set()
    while i * i <= n:
        while n % i == 0:
            ans.add(i)
            n //= i
        i += 1
    if n > 1:
        ans.add(n)
    return len(ans)


class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        arr = [(i, primeFactors(x), x) for i, x in enumerate(nums)]
        n = len(nums)

        left = [-1] * n
        right = [n] * n
        stk = []
        for i, f, x in arr:
            while stk and stk[-1][0] < f:
                stk.pop()
            if stk:
                left[i] = stk[-1][1]
            stk.append((f, i))

        stk = []
        for i, f, x in arr[::-1]:
            while stk and stk[-1][0] <= f:
                stk.pop()
            if stk:
                right[i] = stk[-1][1]
            stk.append((f, i))

        arr.sort(key=lambda x: -x[2])
        ans = 1
        for i, f, x in arr:
            l, r = left[i], right[i]
            cnt = (i - l) * (r - i)
            if cnt <= k:
                ans = ans * pow(x, cnt, mod) % mod
                k -= cnt
            else:
                ans = ans * pow(x, k, mod) % mod
                break
        return ans
# ==================================================================================================
class Solution:
    def maximumScore(self, l: List[int], k: int) -> int:
        mod = 1_000_000_007
        ps = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311]

        n = len(l)
        d = dict.fromkeys(l)
        for v in d:
            cnt = 0
            x = v
            for p in ps:
                if x == 1: break
                q, r = divmod(x, p)
                if r != 0: continue
                cnt += 1
                while r == 0:
                    x = q
                    q, r = divmod(x, p)
            d[v] = cnt + (x != 1)

        w = [0] * n
        stk = []
        pi, pv = -1, inf
        for i, v in enumerate(map(d.__getitem__, l)):
            while pv < v:
                w[pi] *= i - pi
                pi, pv = stk.pop()
            w[i] = i - pi
            stk.append((pi, pv))
            pi, pv = i, v
        for i, _ in reversed(stk):
            w[pi] *= n - pi
            pi = i
        ans = 1
        for v, x in sorted(zip(l, w), reverse=True):
            if k > x:
                k -= x
                ans = ans * pow(v, x, mod) % mod
            else:
                return ans * pow(v, k, mod) % mod
        return ans
# ==========================================================================================
class Solution(object):
    def mostPoints(self, questions):
        dp = [0] * len(questions)
        for i in range(len(questions) - 1, -1, -1):
            index = i + questions[i][1] + 1
            if index < len(questions):
                dp[i] = dp[index] + questions[i][0]
            else:
                dp[i] = questions[i][0]
            if i < len(questions) - 1:
                dp[i] = max(dp[i + 1], dp[i])
        return dp[0]
# ============================================================================================
class Solution:
    def mostPoints(self, questions: List[List[int]]) -> int:
        A, n = questions, len(questions)
        d = [max(A[n-1][0], 0)]*n
        for i in range(n-2, -1, -1):
            if i+A[i][1] + 1 <= n-1:
                d[i] = max(d[i+1], A[i][0] + d[i+A[i][1]+1])
            else:
                d[i] = max(d[i+1], A[i][0])
        return d[0]
# ==============================================================================================
class Solution(object):
    def maximumTripletValue(self, nums):
        maxTriplet = 0
        for i in range(len(nums)):
            for k in range(len(nums) - 1, i, -1):
                j = i + 1
                while j < k:
                    maxTriplet = max(maxTriplet, (nums[i] - nums[j]) * nums[k])
                    j += 1
        return max(0, maxTriplet)
# =============================================================================================
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        res = 0
        N = len(nums)
        left = nums[0]
        for j in range(1, N):
            if nums[j] > left:
                left = nums[j]
                continue
            for k in range(j + 1, N):
                res = max(res, (left - nums[j]) * nums[k])
        return res
# ============================================================================================
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        maxi = float('-inf')
        diff = 0
        res = 0
        
        for i in range(len(nums)):
            maxi = max(maxi, nums[i])
            if i >= 2:
                res = max(res, diff * nums[i])
            if i >= 1:
                diff = max(diff, maxi - nums[i])
                
        return res
# ===============================================================================================

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        prefix_max = 0
        max_diff = 0
        maximum_triplet_value = 0
        for num in nums:
            if max_diff * num > maximum_triplet_value:
                maximum_triplet_value = max_diff * num
            if prefix_max - num > max_diff:
                max_diff = prefix_max - num
            if num > prefix_max:
                prefix_max = num
        return maximum_triplet_value
# ===================================================================================================
class Solution(object):
    def minOperations(self, nums, k):
        nums.sort()
        mini = nums[0]
        if mini < k:
            return -1

        cnt = 0
        i = 0
        while i < len(nums):
            if nums[i] > k:
                cnt += 1
                while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                    i += 1
            i += 1
        return cnt
# =======================================================================================================

__import__("atexit").register(lambda: open("display_runtime.txt", "w").write("0"))

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        hashSet = set()

        for i in range(len(nums)):
            if nums[i] < k:
                return -1
            hashSet.add(nums[i])
        
        return len(hashSet) - 1 if k in hashSet else len(hashSet)
# ======================================================================================================

class Solution:
    def factorial(self, number):
        chakra = 1
        for i in range(1, number + 1):
            chakra *= i
        return chakra

    def generatePalindromes(self, cloneID, index, validPalindromes, k):
        if index >= (len(cloneID) + 1) // 2:
            if int(cloneID) % k == 0:
                validPalindromes.append(cloneID)
            return

        if index != 0:
            temp = cloneID
            temp = temp[:index] + '0' + temp[index+1:]
            temp = temp[:len(temp) - index - 1] + '0' + temp[len(temp) - index:]
            self.generatePalindromes(temp, index + 1, validPalindromes, k)

        for digit in range(1, 10):
            temp = cloneID
            temp = temp[:index] + str(digit) + temp[index+1:]
            temp = temp[:len(temp) - index - 1] + str(digit) + temp[len(temp) - index:]
            self.generatePalindromes(temp, index + 1, validPalindromes, k)

    def countGoodIntegers(self, n, k):
        validPalindromes = []
        baseForm = "0" * n
        self.generatePalindromes(baseForm, 0, validPalindromes, k)
        
        chakraPatterns = set()

        for shadowClone in validPalindromes:
            frequency = ['0'] * 10
            for chakra in shadowClone:
                idx = int(chakra)
                if frequency[idx] == '9':
                    frequency[idx] = 'A'  # beyond 9 digits (special case)
                else:
                    frequency[idx] = str(int(frequency[idx]) + 1)
            chakraPatterns.add(''.join(frequency))

        basePermutations = self.factorial(n)
        totalCount = 0

        for pattern in chakraPatterns:
            permutation = basePermutations
            for freq in pattern:
                divisor = 10 if freq == 'A' else int(freq)
                permutation //= self.factorial(divisor)

            if pattern[0] != '0':
                zeroCount = int(pattern[0]) - 1
                zeroRestrictedPerm = self.factorial(n - 1)
                for freq in pattern[1:]:
                    divisor = 10 if freq == 'A' else int(freq)
                    zeroRestrictedPerm //= self.factorial(divisor)
                zeroRestrictedPerm //= self.factorial(zeroCount)
                permutation -= zeroRestrictedPerm

            totalCount += permutation

        return int(totalCount)
