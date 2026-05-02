import streamlit as st
import pandas as pd
import psycopg2
import requests
from sklearn.ensemble import RandomForestClassifier
import itertools

# ================= 1. 页面 UI 初始化 =================
st.set_page_config(page_title="AI 足球量化投注看板", page_icon="⚽", layout="wide")
st.title("⚽ 智能足球量化投注看板 (全向精算版)")
st.markdown("结合 **Elo 战力模型** 与实时盘口，全向评估【胜/平/负】价值。串关推荐已优化为**胜率优先**排序。")

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
league_options = [
    "soccer_epl (英超)", "soccer_spain_la_liga (西甲)", "soccer_germany_bundesliga (德甲)", 
    "soccer_italy_serie_a (意甲)", "soccer_france_ligue_one (法甲)", "soccer_uefa_champs_league (欧冠)", 
    "soccer_netherlands_eredivisie (荷甲)", "soccer_portugal_primeira_liga (葡超)"
]

selected_league = st.sidebar.selectbox("选择要侦察的联赛:", ["🌟 全部核心联赛 (一键扫描)"] + league_options)

if st.sidebar.button("🚀 一键提取国际盘高价值比赛"):
    st.session_state['predict_clicked'] = True
    st.session_state['matches_data'] = []
    
    if "全部核心联赛" in selected_league:
        leagues_to_fetch = [l.split(" ")[0] for l in league_options]
        st.sidebar.warning(f"全盘扫描将一次性消耗 {len(leagues_to_fetch)} 次 API 额度。")
    else:
        leagues_to_fetch = [selected_league.split(" ")[0]]
    
    my_bar = st.progress(0, text="正在连线欧洲博彩公司...")
    all_fetched_matches = []
    for i, l_key in enumerate(leagues_to_fetch):
        url = f"https://api.the-odds-api.com/v4/sports/{l_key}/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
        resp = requests.get(url)
        if resp.status_code == 200:
            all_fetched_matches.extend(resp.json())
        my_bar.progress((i + 1) / len(leagues_to_fetch), text=f"正在拉取 {l_key} 数据...")
    
    my_bar.empty()
    st.session_state['matches_data'] = all_fetched_matches

# --- 核心优化区：三向评估 (胜/平/负) ---
if st.session_state.get('predict_clicked', False) and st.session_state.get('matches_data') is not None:
    st.subheader("📡 第一阶段：国际盘价值初筛 (胜/平/负全向扫描)")
    matches = st.session_state['matches_data']
    
    intl_val_matches = []
    for match in matches:
        home_team, away_team = match['home_team'], match['away_team']
        try:
            bookmaker = match['bookmakers'][0]
            outcomes = bookmaker['markets'][0]['outcomes']
            
            # 安全提取三个选项的赔率
            home_odds = next((item['price'] for item in outcomes if item['name'] == home_team), 0)
            away_odds = next((item['price'] for item in outcomes if item['name'] == away_team), 0)
            draw_odds = next((item['price'] for item in outcomes if item['name'] == 'Draw'), 0)
        except: continue 
        
        # AI 计算三向胜率
        ht_elo, at_elo = elo_dict.get(home_team, 1500), elo_dict.get(away_team, 1500)
        features = pd.DataFrame({'home_elo': [ht_elo], 'away_elo': [at_elo], 'elo_diff': [ht_elo - at_elo]})
        probs = model.predict_proba(features)[0]
        
        prob_home = probs[list(classes).index('HomeWin')]
        prob_away = probs[list(classes).index('AwayWin')]
        prob_draw = probs[list(classes).index('Draw')]
        
        # 计算三向 EV
        ev_home = calculate_ev(prob_home, home_odds) if home_odds else -1
        ev_away = calculate_ev(prob_away, away_odds) if away_odds else -1
        ev_draw = calculate_ev(prob_draw, draw_odds) if draw_odds else -1
        
        # 挑选出最有价值的选项 (EV最大且大于0的选项)
        best_ev = 0
        best_choice = None
        
        if ev_home > 0 and ev_home > best_ev:
            best_ev, best_choice = ev_home, {'label': f"{home_team} 胜", 'prob': prob_home, 'odds': home_odds}
        if ev_away > 0 and ev_away > best_ev:
            best_ev, best_choice = ev_away, {'label': f"{away_team} 胜", 'prob': prob_away, 'odds': away_odds}
        if ev_draw > 0 and ev_draw > best_ev:
            best_ev, best_choice = ev_draw, {'label': "平局", 'prob': prob_draw, 'odds': draw_odds}
            
        if best_choice:
            intl_val_matches.append({
                'home_team': home_team, 'away_team': away_team,
                'bookmaker': bookmaker['title'], 
                'target_label': best_choice['label'],
                'target_odds': best_choice['odds'],
                'prob': best_choice['prob'], 'ev': best_ev,
                'unique_key': f"jc_{home_team}_{away_team}_{best_choice['label']}"
            })

    if not intl_val_matches:
        st.info("🤷 目前所选联赛中，没有发现国际盘具备投资价值的比赛。")
    else:
        with st.form("jc_odds_form"):
            st.success(f"🔍 成功截获 {len(intl_val_matches)} 个高价值选项！请在下方填入竞彩赔率 (未开售或放弃请填0)：")
            
            for vm in intl_val_matches:
                col1, col2, col3, col4 = st.columns([2, 1, 1.5, 2])
                col1.markdown(f"**⚽ {vm['home_team']}** vs {vm['away_team']} <br> 🎯 **推荐: {vm['target_label']}**", unsafe_allow_html=True)
                col2.markdown(f"国际初盘: `{vm['target_odds']}`")
                col3.markdown(f"AI胜率: `{vm['prob']*100:.1f}%` <br>国际EV: <span style='color:#2e7d32'>+{vm['ev']*100:.1f}%</span>", unsafe_allow_html=True)
                
                default_jc = max(1.01, float(vm['target_odds']) - 0.20)
                col4.number_input(f"🇨🇳 竞彩【{vm['target_label']}】赔率", min_value=0.00, step=0.01, value=default_jc, key=vm['unique_key'])
                st.divider()
                
            submitted = st.form_submit_button("⚙️ 确认以上赔率，生成竞彩【串关精算报告】", type="primary")

        # ================= 5. 竞彩串关精算与避税模块 (胜率优先排序) =================
        valid_jc_matches = []
        for vm in intl_val_matches:
            jc_odds = st.session_state.get(vm['unique_key'], max(1.01, float(vm['target_odds']) - 0.20))
            jc_ev = calculate_ev(vm['prob'], jc_odds)
            
            if jc_ev > 0:
                valid_jc_matches.append({
                    'match_name': vm['target_label'],
                    'prob': vm['prob'], 'odds': jc_odds, 'ev': jc_ev
                })

        if len(valid_jc_matches) >= 2:
            st.header("🔗 竞彩串关智能精算 (Top 20 稳健榜)")
            apply_tax = st.toggle("💸 模拟大额中奖扣税 (奖金超3000强制扣20%)")
            tax_multiplier = 0.8 if apply_tax else 1.0
            
            col_2, col_3 = st.columns(2)
            
            with col_2:
                st.subheader("🔥 稳健 2串1")
                combos_2 = []
                for combo in itertools.combinations(valid_jc_matches, 2):
                    c_prob = combo[0]['prob'] * combo[1]['prob']
                    c_odds = combo[0]['odds'] * combo[1]['odds']
                    eff_odds = c_odds * tax_multiplier
                    c_ev = calculate_ev(c_prob, eff_odds)
                    if c_ev > 0: combos_2.append({'combo': combo, 'prob': c_prob, 'odds': c_odds, 'eff_odds': eff_odds, 'ev': c_ev})
                
                # 排序逻辑修改：首要按概率(prob)降序，次要按收益(ev)降序
                combos_2.sort(key=lambda x: (x['prob'], x['ev']), reverse=True)
                for c in combos_2[:20]:
                    with st.container(border=True):
                        st.markdown(f"**[1]** {c['combo'][0]['match_name']} <br>**[2]** {c['combo'][1]['match_name']}", unsafe_allow_html=True)
                        st.caption(f"综合胜率: {c['prob']*100:.1f}% | 原始赔率: {c['odds']:.2f}")
                        if apply_tax: st.warning(f"🧾 税后赔率: {c['eff_odds']:.2f} | **税后 EV: +{c['ev']*100:.2f}%**")
                        else: st.success(f"📈 **预期收益 (EV): +{c['ev']*100:.2f}%**")
                if not combos_2: st.error("当前条件下，无盈利 2串1 组合。")

            with col_3:
                st.subheader("🚀 稳健 3串1")
                if len(valid_jc_matches) >= 3:
                    combos_3 = []
                    for combo in itertools.combinations(valid_jc_matches, 3):
                        c_prob = combo[0]['prob'] * combo[1]['prob'] * combo[2]['prob']
                        c_odds = combo[0]['odds'] * combo[1]['odds'] * combo[2]['odds']
                        eff_odds = c_odds * tax_multiplier
                        c_ev = calculate_ev(c_prob, eff_odds)
                        if c_ev > 0: combos_3.append({'combo': combo, 'prob': c_prob, 'odds': c_odds, 'eff_odds': eff_odds, 'ev': c_ev})
                    
                    # 排序逻辑修改：首要按概率(prob)降序，次要按收益(ev)降序
                    combos_3.sort(key=lambda x: (x['prob'], x['ev']), reverse=True)
                    for c in combos_3[:20]:
                        with st.container(border=True):
                            st.markdown(f"**[1]** {c['combo'][0]['match_name']} <br>**[2]** {c['combo'][1]['match_name']} <br>**[3]** {c['combo'][2]['match_name']}", unsafe_allow_html=True)
                            st.caption(f"综合胜率: {c['prob']*100:.1f}% | 原始赔率: {c['odds']:.2f}")
                            if apply_tax: st.warning(f"🧾 税后赔率: {c['eff_odds']:.2f} | **税后 EV: +{c['ev']*100:.2f}%**")
                            else: st.success(f"📈 **预期收益 (EV): +{c['ev']*100:.2f}%**")
                    if not combos_3: st.error("当前条件下，无盈利 3串1 组合。")
