# 基于本地知识库的问答生成
## Demo
原理类似[langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM/tree/master)中所展示的流程图

具体实现上，采用UnstructuredFileLoader读入文档，文档支持txt，pdf，docs等数据格式
读入文档后，采用CharacterTextSplitter进行切分

Embedding选择HuggingFaceEmbeddings
- SentenceTransformer不支持arm架构
- openai Embedding 需要消耗tokens

向量数据库储存和查找相似值采用FAISS
- awadb所需格式不匹配

随后采用Azure openai serves得到答案
- 也可以用FastChat或者ChatGLM-6B

### 样例
样例路径: ./data
样例场景为一个淘宝商家的问答bot
用户输入: 这款衣服的袖型是什么
system prompt: 你是一个根据提供的材料作出回答的助手
user prompt: ./data/test-prompt.txt
Output: 这款衣服的袖型是收口袖。