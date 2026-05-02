import streamlit as st
import pandas as pd
import psycopg2
import requests
from sklearn.ensemble import RandomForestClassifier
import itertools # 新增：用于计算串关排列组合

# ================= 1. 页面 UI 初始化 =================
st.set_page_config(page_title="AI 足球量化投注看板", page_icon="⚽", layout="wide")
st.title("⚽ 智能足球量化投注看板 (全欧联赛+竞彩精算版)")
st.markdown("通过结合 **Elo 战力模型** 与 **The Odds API 实时盘口**，自动寻找价值投注，并支持竞彩 2串1/3串1 与扣税收益重估。")

# 从云端安全环境中读取秘密 Key
DB_URI = st.secrets["DB_URI"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]

# ================= 2. 核心算法与模型缓存 =================
@st.cache_resource(show_spinner="⏳ 正在加载历史数据并唤醒 AI 大脑，请稍候...")
def load_and_train_model():
    conn = psycopg2.connect(DB_URI)
    query = "SELECT competition, match_date, home_team, away_team, home_score, away_score FROM matches ORDER BY match_date ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    def get_expected_score(ra, rb): return 1 / (1 + 10 ** ((rb - ra) / 400))
    elo_dict = {}
    home_elo, away_elo = [], []
    
    for _, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        if ht not in elo_dict: elo_dict[ht] = 1500
        if at not in elo_dict: elo_dict[at] = 1500
            
        cur_ht_elo, cur_at_elo = elo_dict[ht], elo_dict[at]
        home_elo.append(cur_ht_elo)
        away_elo.append(cur_at_elo)
        
        ea = get_expected_score(cur_ht_elo, cur_at_elo)
        actual_a = 1 if row['home_score'] > row['away_score'] else (0.5 if row['home_score'] == row['away_score'] else 0)
        elo_dict[ht] = cur_ht_elo + 20 * (actual_a - ea)
        elo_dict[at] = cur_at_elo + 20 * ((1 - actual_a) - (1 - ea))
        
    df['home_elo'] = home_elo
    df['away_elo'] = away_elo
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    
    def get_outcome(row):
        if row['home_score'] > row['away_score']: return 'HomeWin'
        elif row['home_score'] == row['away_score']: return 'Draw'
        else: return 'AwayWin'
    df['outcome'] = df.apply(get_outcome, axis=1)

    X = df[['home_elo', 'away_elo', 'elo_diff']]
    y = df['outcome']
    
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    
    return model, model.classes_, elo_dict

model, classes, elo_dict = load_and_train_model()

# ================= 3. 博彩数学计算器 =================
def calculate_ev(prob, odds): return (prob * odds) - 1

# ================= 4. 实时盘口侦察与页面展示 =================
st.sidebar.header("⚙️ 侦察设置")

# 定义支持的联赛列表
league_options = [
    "soccer_epl (英超)", 
    "soccer_spain_la_liga (西甲)", 
    "soccer_germany_bundesliga (德甲)", 
    "soccer_italy_serie_a (意甲)", 
    "soccer_france_ligue_one (法甲)", 
    "soccer_uefa_champs_league (欧冠)", 
    "soccer_netherlands_eredivisie (荷甲)", 
    "soccer_portugal_primeira_liga (葡超)", 
    "soccer_efl_champ (英冠)"
]

# 新增：一键扫描所有联赛选项
selected_league = st.sidebar.selectbox("选择要侦察的联赛:", ["🌟 全部核心联赛 (一键扫描)"] + league_options)

if st.sidebar.button("🚀 一键预测今日赛事"):
    st.session_state['predict_clicked'] = True
    st.session_state['matches_data'] = []
    
    # 判断是单联赛还是全联赛
    if "全部核心联赛" in selected_league:
        leagues_to_fetch = [l.split(" ")[0] for l in league_options]
        st.sidebar.warning("注意：全盘扫描将一次性消耗 9 次 API 请求额度。")
    else:
        leagues_to_fetch = [selected_league.split(" ")[0]]
    
    # 进度条设计
    progress_text = "正在连线欧洲博彩公司..."
    my_bar = st.progress(0, text=progress_text)
    
    all_fetched_matches = []
    for i, l_key in enumerate(leagues_to_fetch):
        url = f"https://api.the-odds-api.com/v4/sports/{l_key}/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
        resp = requests.get(url)
        if resp.status_code == 200:
            all_fetched_matches.extend(resp.json())
        my_bar.progress((i + 1) / len(leagues_to_fetch), text=f"正在拉取 {l_key} 数据...")
    
    my_bar.empty()
    st.session_state['matches_data'] = all_fetched_matches

# 只要点击过按钮，且缓存里有数据，执行以下渲染逻辑
if st.session_state.get('predict_clicked', False) and st.session_state.get('matches_data') is not None:
    st.subheader("📡 实时盘口侦察报告 (单场分析)")
    matches = st.session_state['matches_data']
    
    # 建立一个“购物车”，专门存放竞彩验证依然 EV>0 的比赛
    valid_jc_matches = []
    
    if not matches:
        st.info("🤷 目前所选联赛没有即将开打的比赛盘口。")
    else:
        for match in matches:
            home_team, away_team = match['home_team'], match['away_team']
            try:
                bookmaker = match['bookmakers'][0]
                home_odds = next(item['price'] for item in bookmaker['markets'][0]['outcomes'] if item['name'] == home_team)
            except:
                continue 
            
            ht_elo = elo_dict.get(home_team, 1500)
            at_elo = elo_dict.get(away_team, 1500)
            
            features = pd.DataFrame({'home_elo': [ht_elo], 'away_elo': [at_elo], 'elo_diff': [ht_elo - at_elo]})
            probs = model.predict_proba(features)[0]
            home_win_prob = probs[list(classes).index('HomeWin')]
            
            ev = calculate_ev(home_win_prob, home_odds)
            
            # 只显示国际盘有价值的比赛，过滤掉垃圾比赛
            if ev > 0:
                with st.container():
                    st.markdown(f"### ⚽ {home_team} (主) vs {away_team}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label=f"国际初盘 ({bookmaker['title']})", value=f"{home_odds}")
                    col2.metric(label="AI 主胜率", value=f"{home_win_prob*100:.1f}%")
                    col3.success(f"🔥 国际盘有价值 (EV: +{ev*100:.1f}%)")
                    
                    with st.expander("🇨🇳 竞彩真实收益二次验算", expanded=True):
                        unique_key = f"jc_{home_team}_{away_team}"
                        jingcai_odds = st.number_input(
                            f"👉 输入体彩店【{home_team} 胜】赔率：", 
                            min_value=1.01, step=0.01, 
                            value=max(1.01, float(home_odds) - 0.20), 
                            key=unique_key
                        )
                        jc_ev = calculate_ev(home_win_prob, jingcai_odds)
                        
                        if jc_ev > 0:
                            st.success(f"✅ 竞彩单关合格 (EV: +{jc_ev*100:.2f}%)，已自动加入下方串关精算池！")
                            # 将合格的比赛加入购物车
                            valid_jc_matches.append({
                                'match_name': f"{home_team} 胜",
                                'prob': home_win_prob,
                                'odds': jingcai_odds
                            })
                        else:
                            st.error(f"❌ 竞彩抽水过高，放弃 (EV: {jc_ev*100:.2f}%)")
                    st.divider()

    # ================= 5. 竞彩串关精算与避税模块 =================
    if len(valid_jc_matches) >= 2:
        st.header("🔗 竞彩串关智能精算 (自动组合验证池)")
        st.markdown("以下组合均由上方**竞彩验证合格**的单场比赛自动交叉组合而成。")
        
        # 扣税开关
        apply_tax = st.toggle("💸 模拟大额中奖扣税 (若单注奖金超过3000元，强制扣除20%所得税)")
        tax_multiplier = 0.8 if apply_tax else 1.0
        
        col_2, col_3 = st.columns(2)
        
        # --- 2串1 分析 ---
        with col_2:
            st.subheader("🔥 推荐 2串1 组合")
            combos_2 = list(itertools.combinations(valid_jc_matches, 2))
            valid_2_count = 0
            
            for combo in combos_2:
                combo_prob = combo[0]['prob'] * combo[1]['prob']
                combo_odds = combo[0]['odds'] * combo[1]['odds']
                
                # 扣税计算
                effective_odds = combo_odds * tax_multiplier
                combo_ev = calculate_ev(combo_prob, effective_odds)
                
                if combo_ev > 0:
                    valid_2_count += 1
                    with st.container(border=True):
                        st.markdown(f"**[1]** {combo[0]['match_name']}  \n**[2]** {combo[1]['match_name']}")
                        st.markdown(f"**综合胜率**: {combo_prob*100:.1f}% | **综合赔率**: {combo_odds:.2f}")
                        if apply_tax:
                            st.warning(f"🧾 税后赔率: {effective_odds:.2f} | **税后 EV**: +{combo_ev*100:.2f}%")
                        else:
                            st.success(f"📈 **预期收益 (EV)**: +{combo_ev*100:.2f}%")
                            
            if valid_2_count == 0:
                st.error("开启扣税后，没有任何 2串1 组合具备投资价值！请降低预期或减小注水避免超 3000 元。")

        # --- 3串1 分析 ---
        with col_3:
            st.subheader("🚀 推荐 3串1 组合")
            if len(valid_jc_matches) >= 3:
                combos_3 = list(itertools.combinations(valid_jc_matches, 3))
                valid_3_count = 0
                
                for combo in combos_3:
                    combo_prob = combo[0]['prob'] * combo[1]['prob'] * combo[2]['prob']
                    combo_odds = combo[0]['odds'] * combo[1]['odds'] * combo[2]['odds']
                    
                    effective_odds = combo_odds * tax_multiplier
                    combo_ev = calculate_ev(combo_prob, effective_odds)
                    
                    if combo_ev > 0:
                        valid_3_count += 1
                        with st.container(border=True):
                            st.markdown(f"**[1]** {combo[0]['match_name']}  \n**[2]** {combo[1]['match_name']}  \n**[3]** {combo[2]['match_name']}")
                            st.markdown(f"**综合胜率**: {combo_prob*100:.1f}% | **综合赔率**: {combo_odds:.2f}")
                            if apply_tax:
                                st.warning(f"🧾 税后赔率: {effective_odds:.2f} | **税后 EV**: +{combo_ev*100:.2f}%")
                            else:
                                st.success(f"📈 **预期收益 (EV)**: +{combo_ev*100:.2f}%")
                
                if valid_3_count == 0:
                    st.error("开启扣税后，没有任何 3串1 组合具备投资价值！")
            else:
                st.info("合格的单场比赛不足 3 场，无法生成 3串1 组合。")
