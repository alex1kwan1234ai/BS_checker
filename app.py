import os
import streamlit as st
from google import genai
from pinecone import Pinecone

# 設定介面預設收起側邊欄，讓主畫面更寬敞
st.set_page_config(page_title="屋宇裝備多功能智能審查員", layout="centered", initial_sidebar_state="collapsed")
st.title("屋宇裝備文件智能審查員")
st.caption("🚀 雲端向量數據庫")

# 1.1 讀取 Gemini API Key
gemini_api_key = st.secrets.get("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ 未偵測到 GEMINI_API_KEY，請在 Streamlit 後台設定。")
    st.stop()

# 1.2 讀取 Pinecone API Key
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
if not pinecone_api_key:
    st.error("❌ 未偵測到 PINECONE_API_KEY，請在 Streamlit 後台設定。")
    st.stop()   

# 初始化雙端 Client
client = genai.Client(api_key=gemini_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("spec") # 連接你網頁上的 spec index

# 定義系統指令
SYSTEM_INSTRUCTION = """
你是一位熟讀香港機電署《電力工作守則》、建築署 ASD General Spec、各項目 Project Spec 以及消防處消防紅皮書的註冊屋宇裝備工程師。
你必須非常仔細地閱讀用戶提供給你的所有參考條文片段。請字斟句酌，不要輕易漏掉任何隱藏在表格、附錄或段落中的數字、年期或規例細節。

【核心強制指令：必須提供原文】
1. 當用戶查詢特定規例或要求時，你必須直接由文件中抄錄（Copy & Paste）相關的「英文原條文」出來，不能只作口頭總結。
2. 抄錄完英文原條文後，請清晰指出你的依據（例如：根據 Project Spec 第 X 條、ASD Spec 第 Y 章或 CoP 守則 Z 及頁數）。
3. 喺列出英文原條文之後，你可以用專業地道的香港地盤廣東話為用戶提供簡短的行內重點解釋。
"""

# =================【2. 主畫面頂部控制區】=================
st.write("### 🛠️ 範疇配置")
col1, col2 = st.columns([3, 1])

with col1:
    task_mode = st.selectbox(
        "請選擇今日要查閱的工程範疇：",
        ["建築署一般指引 (Asd_spec)", "電力安全審查 (Elec_cop)", "消防紅皮書 (Fsd_cop)"],
        label_visibility="collapsed"
    )

# 根據選單動態映射到你上傳時設定的 metadata 標籤（請確保同你上傳時寫入嘅 category 一致）
category_mapping = {
    "建築署一般指引 (Asd_spec)": "asd_spec",
    "電力安全審查 (Elec_cop)": "elec_cop",
    "消防紅皮書 (Fsd_cop)": "fsd_cop"
}
target_category = category_mapping[task_mode]

# 初始化 Session State 狀態記憶體
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_loaded_mode" not in st.session_state:
    st.session_state.current_loaded_mode = None

with col2:
    load_clicked = st.button("⚡ 開始切換", type="primary", use_container_width=True)

# =================【3. 切換邏輯】=================
if load_clicked:
    st.session_state.current_loaded_mode = task_mode
    st.session_state.messages = [] # 切換範疇時自動清空歷史對話
    st.success(f"🎉 成功切換至【{task_mode}】雲端數據庫！")

st.write("---")

# =================【4. 渲染聊天紀錄】=================
if st.session_state.current_loaded_mode == task_mode:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # =================【5. 對話與雲端向量檢索核心邏輯】=================
    if user_query := st.chat_input("想問關於 Spec 嘅咩？"):
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("AI正從雲端進行大範圍關鍵條文檢索..."):            
                # 5.1 調用 Gemini 將用戶提問轉化為 3072 維度向量 (對齊 text-embedding-004)
                query_resp = client.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=[user_query]
                )
                
                # 剥殼語法提煉出純 list 向量
                if isinstance(query_resp.embeddings, list) and len(query_resp.embeddings) > 0:
                    first_emb = query_resp.embeddings[0]
                    raw_vec = first_emb.values if hasattr(first_emb, 'values') else first_emb
                elif hasattr(query_resp.embeddings, 'values'):
                    raw_vec = query_resp.embeddings.values
                else:
                    raw_vec = query_resp.embeddings
                
                query_vector = [float(x) for x in raw_vec]
                
                # 5.2 直接向雲端 Pinecone 發起 Filter 檢索
                query_results = index.query(
                    vector=query_vector,
                    top_k=50, # 保留你最滿意嘅 [-50:] 深度擴容檢索
                    include_metadata=True,
                    filter={"category": {"$eq": target_category}} # 精準篩選指定範疇
                )
                
                # 5.3 拼接撈出來的 50 段條文
                context_pieces = []
                for match in query_results.get("matches", []):
                    meta = match.get("metadata", {})
                    file_src = meta.get("file_name", "參考文件")
                    page_num = int(meta.get("page", 0)) + 1
                    text_content = meta.get("text", "")
                    context_pieces.append(f"[文件來源: {file_src} | 第 {page_num} 頁]:\n{text_content}")
                
                context_text = "\n\n".join(context_pieces)

            with st.spinner("AI正在生成廣東話解釋..."):
                # 5.4 組裝最終的 Prompt 送給 Gemini 生成精準回答
                full_prompt = f"{SYSTEM_INSTRUCTION}\n\n【從文件中精準撈出的相關條文如下】:\n{context_text}\n\n【用戶提問】:\n{user_query}"
                
                response_stream = client.models.generate_content_stream(
                    model="models/gemini-3.8-flash", 
                    contents=full_prompt
                )
                
                # 渲染打字機動態效果
                answer = st.write_stream(chunk.text for chunk in response_stream)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("💡 溫馨提示：請先在上方選擇工程範疇，並點擊右側的『⚡ 開始切換』按鈕激活審查員。")
