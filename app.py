import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model_loader import AttentionModelLoader
from transformers import logging
logging.set_verbosity_error()
# 页面配置
st.set_page_config(
    page_title="🔍 Attention可视化工具",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        color: #FF6B6B;
        text-align: center;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0.2em;
    }
    .subtitle {
        color: #4A5568;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    [data-testid="stSidebar"] {
        background-color: #f7f8fb;
    }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #edf0f5;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(27, 39, 51, 0.06);
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 Attention可视化工具 v1.0</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">交互式理解 Transformer 的 Attention 机制</div>', unsafe_allow_html=True)


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """创建下三角causal mask。"""
    return torch.tril(torch.ones(seq_len, seq_len))


def build_causal_mask_example(seq_len: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    mask = create_causal_mask(seq_len)
    scores = torch.randn(seq_len, seq_len, generator=generator)
    masked_scores = scores.masked_fill(mask == 0, float("-inf"))
    attention = torch.nn.functional.softmax(masked_scores, dim=-1).nan_to_num(0)
    return mask.numpy(), scores.numpy(), masked_scores.numpy(), attention.numpy()


def draw_heatmap(z, title: str, colorscale: str = "Viridis", showscale: bool = True):
    seq_len = len(z)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(range(seq_len)),
            y=list(range(seq_len)),
            colorscale=colorscale,
            showscale=showscale,
            hovertemplate="Query %{y} -> Key %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Key position",
        yaxis_title="Query position",
        height=420,
    )
    return fig

# ============================================
# 侧边栏：模型和输入
# ============================================
loader = AttentionModelLoader()
with st.sidebar:
    st.markdown("## ⚙️ 配置")
    page = st.radio(
        "页面",
        ["Attention 可视化", "Causal Mask 演示"]
    )

if page == "Causal Mask 演示":
    st.subheader("🔒 Causal Mask演示")
    st.info("""
在自回归生成中，模型生成第 t 个 token 时只能看到当前位置和过去位置。
Causal Mask 会把未来位置的 attention score 设为 -inf，softmax 后对应权重变为 0。
""")

    with st.sidebar:
        seq_len = st.slider("序列长度", 3, 10, 5)
        seed = st.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)

    mask, scores, masked_scores, attention = build_causal_mask_example(seq_len, seed)
    display_masked_scores = np.where(mask == 0, -10, masked_scores)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Causal Mask**")
        st.plotly_chart(
            draw_heatmap(mask, "Mask Pattern", "RdYlGn", False),
            use_container_width=True
        )
    with col2:
        st.write("**Before Masking**")
        st.plotly_chart(
            draw_heatmap(scores, "Original Scores"),
            use_container_width=True
        )
    with col3:
        st.write("**After Masking**")
        st.plotly_chart(
            draw_heatmap(display_masked_scores, "Masked Scores"),
            use_container_width=True
        )

    st.write("**Attention After Softmax**")
    st.plotly_chart(
        draw_heatmap(attention, "Final Attention Weights"),
        use_container_width=True
    )

    st.markdown("""
**关键差异**

| 特性 | BERT (Encoder) | GPT (Decoder) |
|------|---|---|
| 可见范围 | 整个句子 | 只看前面的 tokens |
| Mask | 没有 causal mask | Causal Mask |
| 用途 | 理解 | 生成 |
| Attention 模式 | 全连接 | 下三角 |
""")

    with st.expander("查看矩阵数值"):
        st.write("Causal Mask")
        st.dataframe(mask.astype(int), use_container_width=True)
        st.write("Original Scores")
        st.dataframe(np.round(scores, 4), use_container_width=True)
        st.write("Masked Scores")
        st.dataframe(np.round(display_masked_scores, 4), use_container_width=True)
        st.write("Final Attention")
        st.dataframe(np.round(attention, 4), use_container_width=True)

    st.stop()

with st.sidebar:
    
    model_choice = st.selectbox(
        "选择模型",
        loader.get_available_models()
    )
    info = loader.get_model_info(model_choice)
    st.markdown(f"**模型描述**: {info['description']}")
    
    text_input = st.text_area(
        "输入文本",
        value="The quick brown fox jumps over the lazy dog",
        height=100
    )

# ============================================
# 模型介绍
# ============================================

model_info = loader.get_model_info(model_choice)
st.subheader(f"{model_choice} 模型介绍")
st.write(model_info["intro"])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("模型名称", model_info["name"])
with col2:
    st.metric("结构类型", model_info["type"])
with col3:
    st.metric("最大长度", model_info["max_length"])

st.markdown(f"**模型特点**: {model_info['features']}")
st.markdown(f"**适合场景**: {model_info['best_for']}")

# ============================================
# 加载模型（缓存）
# ============================================

def positional_encoding(d_model: int, max_len: int = 100):
    """生成用于可视化的sin/cos位置编码矩阵"""
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:pe[:, 1::2].shape[1]])

    return pe


@st.cache_resource(show_spinner=False)
def get_model_and_tokenizer(model_key):
    """缓存模型和tokenizer，避免切换控件时重复加载。"""
    cached_loader = AttentionModelLoader()
    return cached_loader.load(model_key)


@st.cache_data(show_spinner=False)
def compute_attention_bundle(text, model_key):
    """缓存同一文本和模型的完整attention结果。"""
    cached_loader = AttentionModelLoader()
    model_info = cached_loader.get_model_info(model_key)
    tokenizer, model = get_model_and_tokenizer(model_key)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=model_info["max_length"]
    )

    with torch.no_grad():
        outputs = model(**inputs)

    if outputs.attentions is None:
        return None

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attentions = [
        layer_attention[0].detach().cpu().numpy()
        for layer_attention in outputs.attentions
    ]
    return tokens, attentions


def compute_attention(text, model_key, layer, head):
    """返回单个layer/head的attention矩阵。"""
    result = compute_attention_bundle(text, model_key)
    if result is None:
        return None
    _, attentions = result
    return attentions[layer][head]

# ============================================
# 计算Attention
# ============================================

if text_input:
    with st.spinner("正在加载模型并计算 Attention..."):
        attention_result = compute_attention_bundle(text_input, model_choice)

    if attention_result is None:
        st.error("当前模型没有返回 attention。")
        st.stop()

    st.success("✅ 模型加载成功，Attention 已计算完成。")

    tokens, attentions = attention_result
    num_layers = len(attentions)
    num_heads = attentions[0].shape[0]

    with st.sidebar:
        layer = st.slider("选择层数", 0, num_layers - 1, num_layers - 1)
        head = st.slider("选择注意力头", 0, num_heads - 1, 0)
        max_heads_to_show = st.slider(
            "多头对比显示数量",
            1,
            num_heads,
            min(4, num_heads)
        )

    if len(tokens) > 512:
        st.warning("⚠️ 序列过长，可能导致可视化变慢。")

    # ============================================
    # Token显示区
    # ============================================

    tokenizer, _ = get_model_and_tokenizer(model_choice)
    special_tokens = [token for token in tokens if token in tokenizer.all_special_tokens]

    st.subheader("当前 token 列表")
    st.code(" ".join(tokens), language="text")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("token 数量", len(tokens))
    with col2:
        if special_tokens:
            st.markdown(f"**特殊 token**: {' '.join(special_tokens)}")
        else:
            st.markdown("**特殊 token**: 当前输入未包含特殊 token")
    
    # 获取选中的layer和head的attention
    attention_matrix = compute_attention(text_input, model_choice, layer, head)
    
    # ============================================
    # 标签页布局
    # ============================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 单头分析", "🎨 多头对比", "🧱 层级对比", "📈 统计信息", "📍 位置编码"])
    
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

        st.subheader("Token 关注分析")
        token_options = [f"{index}: {token}" for index, token in enumerate(tokens)]
        source_index = st.selectbox(
            "选择一个 token，查看它主要关注了哪些 token",
            range(len(tokens)),
            format_func=lambda index: token_options[index]
        )
        top_k = st.slider("显示 Top K", 1, len(tokens), min(5, len(tokens)))

        attention_scores = attention_matrix[source_index]
        top_indices = attention_scores.argsort()[::-1][:top_k]
        top_tokens = [
            {
                "排名": rank,
                "目标位置": int(target_index),
                "目标 token": tokens[target_index],
                "注意力值": float(attention_scores[target_index])
            }
            for rank, target_index in enumerate(top_indices, start=1)
        ]

        st.markdown(f"**{tokens[source_index]}** 最关注的 token：")
        st.dataframe(top_tokens, use_container_width=True)

        fig_top = go.Figure(
            data=go.Bar(
                x=[tokens[index] for index in top_indices],
                y=[attention_scores[index] for index in top_indices]
            )
        )
        fig_top.update_layout(
            title=f"{tokens[source_index]} 的 Top {top_k} 注意力",
            xaxis_title="目标 token",
            yaxis_title="Attention Score",
            height=360
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    # ============================================
    # Tab 2：多头对比
    # ============================================
    
    with tab2:
        st.subheader(f"第{layer}层注意力头对比")
        st.caption(f"当前显示前 {max_heads_to_show} 个头，减少图表数量以提升响应速度。")
        
        # 多头子图
        cols = 4
        rows = (max_heads_to_show + cols - 1) // cols
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Head {i}" for i in range(max_heads_to_show)]
        )
        
        for i in range(max_heads_to_show):
            row = i // cols + 1
            col = i % cols + 1
            
            att_mat = attentions[layer][i]
            
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
            title=f"{max_heads_to_show} Attention Heads - Layer {layer}",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # Tab 3：层级对比
    # ============================================

    with tab3:
        st.subheader(f"同一个注意力头在不同层的变化：Head {head}")

        compare_layers = [0, num_layers // 2, num_layers - 1]
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[f"Layer {layer_index}" for layer_index in compare_layers]
        )

        for col, layer_index in enumerate(compare_layers, start=1):
            layer_attention = attentions[layer_index][head]

            fig.add_trace(
                go.Heatmap(
                    z=layer_attention,
                    x=tokens,
                    y=tokens,
                    colorscale="Viridis",
                    showscale=(col == 3)
                ),
                row=1,
                col=col
            )

        fig.update_layout(
            title=f"Layer 0 / Middle / Last Attention Compare - Head {head}",
            height=520,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # Tab 4：统计信息
    # ============================================
    
    with tab4:
        st.subheader("Attention统计")
        
        # 计算每个头的最大关注值
        max_values = []
        avg_values = []
        for i in range(num_heads):
            att_mat = attentions[layer][i]
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

    # ============================================
    # Tab 5：Position Encoding可视化
    # ============================================

    with tab5:
        st.subheader("Position Encoding可视化")
        st.info("这里仅可视化sin/cos位置编码，不会把位置编码重新传入当前模型。")

        d_model = st.slider("Embedding维度", 64, 512, 256, step=64)
        max_len = st.slider("最大序列长度", 10, 100, 50)

        pe = positional_encoding(d_model, max_len)

        fig_pe = go.Figure(data=go.Heatmap(
            z=pe,
            x=list(range(d_model)),
            y=list(range(max_len)),
            colorscale="Viridis"
        ))

        fig_pe.update_layout(
            title="Position Encoding Matrix",
            xaxis_title="Embedding Dimension",
            yaxis_title="Position",
            height=600
        )

        st.plotly_chart(fig_pe, use_container_width=True)

        st.markdown("""
**观察重点**

- 不同维度使用不同频率的 sin/cos 波。
- 位置越远，编码差异通常越明显。
- 这个图只是帮助理解位置信息，不参与当前模型推理。
""")

else:
    st.info("👈 请在侧边栏输入文本")
