# Q71.) Simplify Path

# You are given an absolute path for a Unix-style file system, which always begins with a slash '/'. Your task is to transform this absolute path into its simplified canonical path.

# The rules of a Unix-style file system are as follows:

# A single period '.' represents the current directory.
# A double period '..' represents the previous/parent directory.
# Multiple consecutive slashes such as '//' and '///' are treated as a single slash '/'.
# Any sequence of periods that does not match the rules above should be treated as a valid directory or file name. For example, '...' and '....' are valid directory or file names.
# The simplified canonical path should follow these rules:

# The path must start with a single slash '/'.
# Directories within the path must be separated by exactly one slash '/'.
# The path must not end with a slash '/', unless it is the root directory.
# The path must not have any single or double periods ('.' and '..') used to denote current or parent directories.
# Return the simplified canonical path.

# Example 1:

# Input: path = "/home/"

# Output: "/home"

# Explanation:

# The trailing slash should be removed.

# Example 2:

# Input: path = "/home//foo/"

# Output: "/home/foo"

# Explanation:

# Multiple consecutive slashes are replaced by a single one.

# Example 3:

# Input: path = "/home/user/Documents/../Pictures"

# Output: "/home/user/Pictures"

# Explanation:

# A double period ".." refers to the directory up a level (the parent directory).

# Example 4:

# Input: path = "/../"

# Output: "/"

# Explanation:

# Going one level up from the root directory is not possible.

# Example 5:

# Input: path = "/.../a/../b/c/../d/./"

# Output: "/.../b/d"

# Explanation:

# "..." is a valid name for a directory in this problem.

# Constraints:

# 1 <= path.length <= 3000
# path consists of English letters, digits, period '.', slash '/' or '_'.
# path is a valid absolute Unix path.

# Sol_71}

class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        n = len(path)
        i = 0
        
        while i < n:
            while i < n and path[i] == '/':
                i += 1
                
            start = i
            while i < n and path[i] != '/':
                i += 1
            dir = path[start:i]
            
            if dir == "..":
                if stack:
                    stack.pop()
            elif dir and dir != ".":
                stack.append(dir)
        
        return '/' + '/'.join(stack)

# Q72.) Edit Distance

# Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

# You have the following three operations permitted on a word:

# Insert a character
# Delete a character
# Replace a character
 

# Example 1:

# Input: word1 = "horse", word2 = "ros"
# Output: 3
# Explanation: 
# horse -> rorse (replace 'h' with 'r')
# rorse -> rose (remove 'r')
# rose -> ros (remove 'e')
# Example 2:

# Input: word1 = "intention", word2 = "execution"
# Output: 5
# Explanation: 
# intention -> inention (remove 't')
# inention -> enention (replace 'i' with 'e')
# enention -> exention (replace 'n' with 'x')
# exention -> exection (replace 'n' with 'c')
# exection -> execution (insert 'u')
 
# Constraints:

# 0 <= word1.length, word2.length <= 500
# word1 and word2 consist of lowercase English letters.

# Sol_72}

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        cache = [[float("inf")]* (len(word2)+1) for _ in range(len(word1)+1)]
        for col in range(len(word2)+1): 
            cache[len(word1)][col] = len(word2) - col 
        for row in range(len(word1)+1): 
            cache[row][len(word2)] = len(word1) - row
        for i in range(len(word1) - 1, -1, -1):
            for j in range(len(word2)-1,-1,-1):
                if word1[i] == word2[j]:
                    cache[i][j] = cache[i+1][j+1]
                else: 
                    cache[i][j] = min(cache[i+1][j],cache[i][j+1],cache[i+1][j+1])+1 

        return cache[0][0]