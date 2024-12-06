# Q101.) Symmetric Tree
# Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

# Input: root = [1,2,2,3,4,4,3]
# Output: true

# Input: root = [1,2,2,null,3,null,3]
# Output: false
 
# Constraints:

# The number of nodes in the tree is in the range [1, 1000].
# -100 <= Node.val <= 100
 
# Follow up: Could you solve it both recursively and iteratively?

# Sol_101}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isMirror(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return left.val == right.val and self.isMirror(left.left, right.right) and self.isMirror(left.right, right.left)
    
    def isSymmetric(self, root):
        if not root:
            return True
        return self.isMirror(root.left, root.right)

# Q102.) Binary Tree Level Order Traversal

# Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

# Input: root = [3,9,20,null,null,15,7]
# Output: [[3],[9,20],[15,7]]

# Input: root = [1]
# Output: [[1]]

# Input: root = []
# Output: []
 
# Constraints:

# The number of nodes in the tree is in the range [0, 2000].
# -1000 <= Node.val <= 1000

# Sol_102}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        tree_q = deque()
        ans = []
        if root: tree_q.append(root)
        while tree_q:
            level_node_list = []
            for i in range(len(tree_q)):
                cur = tree_q.popleft()
                level_node_list.append(cur.val)
                if cur.left: tree_q.append(cur.left)
                if cur.right: tree_q.append(cur.right)
            if level_node_list: ans.append(level_node_list)
        return ans

# Q103.) Binary Tree Zigzag Level Order Traversal

# Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

# Input: root = [3,9,20,null,null,15,7]
# Output: [[3],[20,9],[15,7]]

# Input: root = [1]
# Output: [[1]]

# Input: root = []
# Output: []
 
# Constraints:

# The number of nodes in the tree is in the range [0, 2000].
# -100 <= Node.val <= 100

# Sol_103}

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        b=[root]
        res=[]
        flag=True
        while b:
            val=[]
            b1=[]
            for i in b:
                val.append(i.val)
                if i.left:
                    b1.append(i.left)
                if i.right:
                    b1.append(i.right)
            if flag:
                res.append(val)
                flag=False
            else:
                res.append(val[::-1])
                flag=True
            b=b1
        return res

# Q104.) Maximum Depth of Binary Tree
# Given the root of a binary tree, return its maximum depth.

# A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

# Input: root = [3,9,20,null,null,15,7]
# Output: 3

# Input: root = [1,null,2]
# Output: 2

# Constraints:

# The number of nodes in the tree is in the range [0, 104].
# -100 <= Node.val <= 100

# Sol_104}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        stack = deque([(root, 1)])
        max_depth = 0
        
        while stack:
            node, depth = stack.pop()
            if node:
                max_depth = max(max_depth, depth)
                stack.append((node.left, depth + 1))
                stack.append((node.right, depth + 1))
        
        return max_depth        

# Q105.) Construct Binary Tree from Preorder and Inorder Traversal
# Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

# Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
# Output: [3,9,20,null,null,15,7]

# Input: preorder = [-1], inorder = [-1]
# Output: [-1]

# Constraints:

# 1 <= preorder.length <= 3000
# inorder.length == preorder.length
# -3000 <= preorder[i], inorder[i] <= 3000
# preorder and inorder consist of unique values.
# Each value of inorder also appears in preorder.
# preorder is guaranteed to be the preorder traversal of the tree.
# inorder is guaranteed to be the inorder traversal of the tree.

# Sol_105}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if len(preorder) == 1: return TreeNode(preorder[0])
        preorder = deque(preorder)
        def build(l, h):
            if l > h: return None
            
            curr = preorder.popleft()
            idx = inorder.index(curr)
            root = TreeNode(inorder[idx])

            root.left = build(l, idx-1)
            root.right = build(idx+1, h)

            return root
        
        return build(0, len(inorder)-1)

# Q106.) Construct Binary Tree from Inorder and Postorder Traversal
# Given two integer arrays inorder and postorder where inorder is the inorder traversal of a binary tree and postorder is the postorder traversal of the same tree, construct and return the binary tree.

# Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
# Output: [3,9,20,null,null,15,7]

# Input: inorder = [-1], postorder = [-1]
# Output: [-1]
 
# Constraints:

# 1 <= inorder.length <= 3000
# postorder.length == inorder.length
# -3000 <= inorder[i], postorder[i] <= 3000
# inorder and postorder consist of unique values.
# Each value of postorder also appears in inorder.
# inorder is guaranteed to be the inorder traversal of the tree.
# postorder is guaranteed to be the postorder traversal of the tree.

# Sol_106}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        inorderidx = {v : i for i, v in enumerate(inorder)}

        def helper(l, r):
            if l > r:
                return None
            
            root = TreeNode(postorder.pop())

            idx = inorderidx[root.val]
            root.right = helper(idx + 1, r)
            root.left = helper(l, idx - 1)
            return root
        
        return helper(0, len(inorder) - 1)

# Q107.) Binary Tree Level Order Traversal II
# Given the root of a binary tree, return the bottom-up level order traversal of its nodes' values. (i.e., from left to right, level by level from leaf to root).

# Input: root = [3,9,20,null,null,15,7]
# Output: [[15,7],[9,20],[3]]

# Input: root = [1]
# Output: [[1]]

# Input: root = []
# Output: []
 
# Constraints:

# The number of nodes in the tree is in the range [0, 2000].
# -1000 <= Node.val <= 1000

# Sol_107}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
        q = collections.deque()
        q.append(root)
        while q:
            q_len = len(q)
            level = []
            for i in range(q_len):
                node = q.popleft()
                if node is not None:
                    level.append(node.val)
                    q.append(node.left)
                    q.append(node.right)
            if level:
                res.append(level)

        return res[::-1]

# Q108.) Convert Sorted Array to Binary Search Tree

# Given an integer array nums where the elements are sorted in ascending order, convert it to a 
# height-balanced
#  binary search tree.

# Input: nums = [-10,-3,0,5,9]
# Output: [0,-3,9,-10,null,5]
# Explanation: [0,-10,5,null,-3,null,9] is also accepted:

# Input: nums = [1,3]
# Output: [3,1]
# Explanation: [1,null,3] and [3,1] are both height-balanced BSTs.
 
# Constraints:

# 1 <= nums.length <= 104
# -104 <= nums[i] <= 104
# nums is sorted in a strictly increasing order.

# SOl_108}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        
        root = TreeNode(0) 
        stack = [(root, 0, len(nums) - 1)]
        
        while stack:
            node, left, right = stack.pop()
            mid = left + (right - left) // 2
            node.val = nums[mid]
            
            if left <= mid - 1:
                node.left = TreeNode(0) 
                stack.append((node.left, left, mid - 1))
                
            if mid + 1 <= right:
                node.right = TreeNode(0) 
                stack.append((node.right, mid + 1, right))
        
        return root   

# Q109.) Convert Sorted List to Binary Search Tree
# Given the head of a singly linked list where elements are sorted in ascending order, convert it to a 
# height-balanced
#  binary search tree.

# Input: head = [-10,-3,0,5,9]
# Output: [0,-3,9,-10,null,5]
# Explanation: One possible answer is [0,-3,9,-10,null,5], which represents the shown height balanced BST.

# Input: head = []
# Output: []
 
# Constraints:

# The number of nodes in head is in the range [0, 2 * 104].
# -105 <= Node.val <= 105  

# Sol_109}

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        if head==None:
            return None
        if head.next==None:
            return TreeNode(head.val)
        slow,fast=head,head.next
        while fast.next and fast.next.next:
            slow=slow.next
            fast=fast.next.next
        mid=slow.next
        slow.next=None
        root=TreeNode(mid.val)
        root.left=self.sortedListToBST(head)
        root.right=self.sortedListToBST(mid.next)
        return root

# Q110.) Balanced Binary Tree
# Given a binary tree, determine if it is 
# height-balanced

# Input: root = [3,9,20,null,null,15,7]
# Output: true

# Input: root = [1,2,2,3,3,null,null,4,4]
# Output: false

# Input: root = []
# Output: true
 
# Constraints:

# The number of nodes in the tree is in the range [0, 5000].
# -104 <= Node.val <= 104

# Sol_110}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def checkBalance(node):
            if not node:
                return True, 0
            
            left_balanced, left_height = checkBalance(node.left)
            right_balanced, right_height = checkBalance(node.right)            
            balanced = left_balanced and right_balanced and abs(left_height - right_height) <= 1
           
            height = max(left_height, right_height) + 1
            return balanced, height
        
        balanced, _ = checkBalance(root)
        return balanced

# OR

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return (self.Height(root) >= 0)
    def Height(self, root):
        if root is None:  return 0
        leftheight, rightheight = self.Height(root.left), self.Height(root.right)
        if leftheight < 0 or rightheight < 0 or abs(leftheight - rightheight) > 1:  return -1
        return max(leftheight, rightheight) + 1        

# Q111.) Minimum Depth of Binary Tree
# Given a binary tree, find its minimum depth.

# The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

# Note: A leaf is a node with no children.

# Input: root = [3,9,20,null,null,15,7]
# Output: 2

# Input: root = [2,null,3,null,4,null,5,null,6]
# Output: 5
 
# Constraints:

# The number of nodes in the tree is in the range [0, 105].
# -1000 <= Node.val <= 1000

# Sol_111}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        queue = deque([(root, 1)]) 

        while queue:
            node, depth = queue.popleft()

            if not node.left and not node.right:
                return depth

            if node.left:
                queue.append((node.left, depth + 1))
            if node.right:
                queue.append((node.right, depth + 1))       

# Q112.) Path Sum
# Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

# A leaf is a node with no children.

# Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
# Output: true
# Explanation: The root-to-leaf path with the target sum is shown.

# Input: root = [1,2,3], targetSum = 5
# Output: false
# Explanation: There are two root-to-leaf paths in the tree:
# (1 --> 2): The sum is 3.
# (1 --> 3): The sum is 4.
# There is no root-to-leaf path with sum = 5.

# Input: root = [], targetSum = 0
# Output: false
# Explanation: Since the tree is empty, there are no root-to-leaf paths.
 
# Constraints:

# The number of nodes in the tree is in the range [0, 5000].
# -1000 <= Node.val <= 1000
# -1000 <= targetSum <= 1000

# Sol_112}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        
        if not root.left and not root.right:
            return targetSum == root.val
        
        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)

# Q113} Path Sum II
# Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of the node values in the path equals targetSum. Each path should be returned as a list of the node values, not node references.

# A root-to-leaf path is a path starting from the root and ending at any leaf node. A leaf is a node with no children.

# Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
# Output: [[5,4,11,2],[5,8,4,5]]
# Explanation: There are two paths whose sum equals targetSum:
# 5 + 4 + 11 + 2 = 22
# 5 + 8 + 4 + 5 = 22

# Input: root = [1,2,3], targetSum = 5
# Output: []

# Input: root = [1,2], targetSum = 0
# Output: []
 
# Constraints:

# The number of nodes in the tree is in the range [0, 5000].
# -1000 <= Node.val <= 1000
# -1000 <= targetSum <= 1000

# Sol_113}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        def solve(l,root,ans):
            if root is None:
                return
            l.append(root.val)
            
            if root.left is None and root.right is None:
                if sum(l) == targetSum:
                    ans.append(l)
                return
            x = l[:]
            if root.left:
                solve(l,root.left,ans)
            if root.right:
                solve(x,root.right,ans)

        ans = []
        l = []
        solve(l,root,ans)
        return ans

# Q114.) Flatten Binary Tree to Linked List
# Given the root of a binary tree, flatten the tree into a "linked list":

# The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
# The "linked list" should be in the same order as a pre-order traversal of the binary tree.

# Input: root = [1,2,5,3,4,null,6]
# Output: [1,null,2,null,3,null,4,null,5,null,6]

# Input: root = []
# Output: []

# Input: root = [0]
# Output: [0]
 
# Constraints:

# The number of nodes in the tree is in the range [0, 2000].
# -100 <= Node.val <= 100
 
# Follow up: Can you flatten the tree in-place (with O(1) extra space)?

# Sol_114}

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        if root is None:
            return

        current = root
        while current is not None:

            if current.left is not None:
                last = current.left
                while last.right is not None:
                    last = last.right

                last.right = current.right
                current.right = current.left
                current.left = None

            current = current.right

# Q115.) Distinct Subsequences
# Given two strings s and t, return the number of distinct subsequences of s which equals t.

# The test cases are generated so that the answer fits on a 32-bit signed integer.

# Input: s = "rabbbit", t = "rabbit"
# Output: 3
# Explanation:
# As shown below, there are 3 ways you can generate "rabbit" from s.
# rabbbit
# rabbbit
# rabbbit

# Input: s = "babgbag", t = "bag"
# Output: 5
# Explanation:
# As shown below, there are 5 ways you can generate "bag" from s.
# babgbag
# babgbag
# babgbag
# babgbag
# babgbag
 
# Constraints:

# 1 <= s.length, t.length <= 1000
# s and t consist of English letters.

# Sol_115}

class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        s2 = s
        s1 = t
        d = {}
        for i in range(len(s1)):
            if s1[i] in d:
                d[s1[i]].append(i)
            else:
                d[s1[i]] = [i]
        s = s2[::-1]
        sumlst = [0 for j in range(len(s1))] 
        for i in range(len(s2)):
            if s[i] in d:
                for idx in d[s[i]]:
                    if idx == len(s1)-1:
                        sumlst[idx] +=1
                    else:
                        sumlst[idx] += sumlst[idx+1]
        return sumlst[0]

# Q116.} Populating Next Right Pointers in Each Node
# You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

# struct Node {
#   int val;
#   Node *left;
#   Node *right;
#   Node *next;
# }
# Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

# Initially, all next pointers are set to NULL.

# Input: root = [1,2,3,4,5,6,7]
# Output: [1,#,2,3,#,4,5,6,7,#]
# Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.

# Input: root = []
# Output: []
 
# Constraints:

# The number of nodes in the tree is in the range [0, 212 - 1].
# -1000 <= Node.val <= 1000
 
# Follow-up:

# You may only use constant extra space.
# The recursive approach is fine. You may assume implicit stack space does not count as extra space for this problem.

# Sol_116}

class Solution:
    def connect(self, root):
        if not root: return None
        q = deque([root])
        while q:
            rightNode = None
            for _ in range(len(q)):
                cur = q.popleft()
                cur.next, rightNode = rightNode, cur
                if cur.right:
                    q.extend([cur.right, cur.left])
        return root