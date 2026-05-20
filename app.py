import streamlit as st
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModel

# 页面配置
st.set_page_config(
    page_title="🔍 Attention可视化工具",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Attention可视化工具 v1.0")
st.markdown("**交互式理解Transformer的Attention机制**")

# ============================================
# 侧边栏：模型和输入
# ============================================

with st.sidebar:
    st.markdown("## ⚙️ 配置")
    
    model_choice = st.selectbox(
        "选择模型",
        ["bert-base-uncased", "bert-base-cased"]
    )
    
    text_input = st.text_area(
        "输入文本",
        value="The quick brown fox jumps over the lazy dog",
        height=100
    )

# ============================================
# 加载模型（缓存）
# ============================================

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model(model_choice)

# ============================================
# 计算Attention
# ============================================

if text_input:
    inputs = tokenizer(text_input, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    attentions = outputs.attentions

    if attentions is None:
        st.error("当前模型没有返回 attention。")
        st.stop()

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    with st.sidebar:
        layer = st.slider("选择层数", 0, num_layers - 1, num_layers - 1)
        head = st.slider("选择注意力头", 0, num_heads - 1, 0)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # 获取选中的layer和head的attention
    attention_matrix = attentions[layer][0, head].detach().cpu().numpy()
    
    # ============================================
    # 标签页布局
    # ============================================
    
    tab1, tab2, tab3 = st.tabs(["📊 单头分析", "🎨 多头对比", "📈 统计信息"])
    
    # ============================================
    # Tab 1：单头分析
    # ============================================
    
    with tab1:
        st.subheader(f"第{layer}层 第{head}头的注意力")
        
        # 绘制热力图
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            hovertemplate='From %{y} to %{x}: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Attention Matrix - Layer {layer}, Head {head}",
            xaxis_title="To (Keys)",
            yaxis_title="From (Queries)",
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 分析信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均关注值", f"{attention_matrix.mean():.4f}")
        with col2:
            st.metric("最大关注值", f"{attention_matrix.max():.4f}")
        with col3:
            st.metric("关注集中度", f"{(attention_matrix > 0.5).sum() / attention_matrix.size:.1%}")
    
    # ============================================
    # Tab 2：多头对比
    # ============================================
    
    with tab2:
        st.subheader(f"第{layer}层的所有{num_heads}个注意力头")
        
        # 多头子图
        cols = 4
        rows = (num_heads + cols - 1) // cols
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Head {i}" for i in range(num_heads)]
        )
        
        for i in range(num_heads):
            row = i // cols + 1
            col = i % cols + 1
            
            att_mat = attentions[layer][0, i].detach().cpu().numpy()
            
            fig.add_trace(
                go.Heatmap(
                    z=att_mat,
                    x=tokens,
                    y=tokens,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=320 * rows,
            title=f"All {num_heads} Attention Heads - Layer {layer}",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # Tab 3：统计信息
    # ============================================
    
    with tab3:
        st.subheader("Attention统计")
        
        # 计算每个头的最大关注值
        max_values = []
        avg_values = []
        for i in range(num_heads):
            att_mat = attentions[layer][0, i].detach().cpu().numpy()
            max_values.append(att_mat.max())
            avg_values.append(att_mat.mean())
        
        # 绘制
        fig_max = go.Figure(
            data=go.Bar(x=list(range(num_heads)), y=max_values)
        )
        fig_max.update_layout(
            title="每个头的最大注意力值",
            xaxis_title="Head",
            yaxis_title="Max Attention",
            height=400
        )
        st.plotly_chart(fig_max, use_container_width=True)
        
        fig_avg = go.Figure(
            data=go.Bar(x=list(range(num_heads)), y=avg_values)
        )
        fig_avg.update_layout(
            title="每个头的平均注意力值",
            xaxis_title="Head",
            yaxis_title="Average Attention",
            height=400
        )
        st.plotly_chart(fig_avg, use_container_width=True)

else:
    st.info("👈 请在侧边栏输入文本")
