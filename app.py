import streamlit as st
import pandas as pd
import psycopg2
import requests
from sklearn.ensemble import RandomForestClassifier
import itertools
from datetime import datetime, timedelta, timezone

# ================= 1. 页面 UI 初始化 =================
st.set_page_config(page_title="AI 量化精算看板 V4", page_icon="⚖️", layout="wide")
st.title("⚖️ 智能足球量化投注看板 (防方差精算版)")
st.markdown("已开启**概率熔断机制**与**凯利仓位管理**，强制过滤高风险组合，确保长期稳健。")

DB_URI = st.secrets["DB_URI"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]

# ================= 2. 核心算法与模型缓存 =================
@st.cache_resource(show_spinner="⏳ 唤醒 AI 大脑，加载 Elo 数据...")
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

# 新增：凯利公式计算器 (使用 1/4 凯利防破产策略)
def calculate_kelly(prob, odds, fraction=0.25):
    b = odds - 1
    if b <= 0: return 0
    kelly = (prob * b - (1 - prob)) / b
    return max(0, kelly * fraction)

# ================= 4. 实时盘口侦察与页面展示 =================
st.sidebar.header("⚙️ 严苛风控设置")
league_options = [
    "soccer_epl (英超)", "soccer_spain_la_liga (西甲)", "soccer_germany_bundesliga (德甲)", 
    "soccer_italy_serie_a (意甲)", "soccer_france_ligue_one (法甲)", "soccer_uefa_champs_league (欧冠)", 
    "soccer_netherlands_eredivisie (荷甲)", "soccer_portugal_primeira_liga (葡超)"
]
selected_league = st.sidebar.selectbox("选择联赛:", ["🌟 全部核心联赛 (一键扫描)"] + league_options)

# 核心优化1：增加最低胜率熔断调节
st.sidebar.markdown("---")
st.sidebar.subheader("📉 最低胜率熔断")
st.sidebar.caption("为防止串关出现极低概率事件，过滤掉所有 AI 胜率低于此阈值的赛果。")
min_prob_threshold = st.sidebar.slider("要求 AI 胜率必须 ≥ (%)", min_value=10, max_value=60, value=35, step=1) / 100.0

if st.sidebar.button("🚀 启动量化引擎 (API抓取)"):
    st.session_state['predict_clicked'] = True
    st.session_state['matches_data'] = []
    
    leagues_to_fetch = [l.split(" ")[0] for l in league_options] if "全部核心联赛" in selected_league else [selected_league.split(" ")[0]]
    
    my_bar = st.progress(0, text="正在连线博彩公司...")
    all_fetched_matches = []
    for i, l_key in enumerate(leagues_to_fetch):
        url = f"https://api.the-odds-api.com/v4/sports/{l_key}/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
        resp = requests.get(url)
        if resp.status_code == 200: all_fetched_matches.extend(resp.json())
        my_bar.progress((i + 1) / len(leagues_to_fetch), text=f"拉取 {l_key} ...")
    
    my_bar.empty()
    st.session_state['matches_data'] = all_fetched_matches

# --- 三向评估 + 胜率熔断 + 时间过滤 ---
if st.session_state.get('predict_clicked', False) and st.session_state.get('matches_data') is not None:
    st.subheader("📡 第一阶段：高胜率价值池初筛")
    matches = st.session_state['matches_data']
    
    tz_bj = timezone(timedelta(hours=8))
    now_bj = datetime.now(tz_bj)
    threshold_bj = (now_bj + timedelta(days=2)).replace(hour=23, minute=59, second=59)
    
    intl_val_matches = []
    for match in matches:
        try:
            match_time_utc = datetime.strptime(match['commence_time'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            match_time_bj = match_time_utc.astimezone(tz_bj)
            if match_time_bj > threshold_bj: continue 
            match_time_str = match_time_bj.strftime("%m-%d %H:%M")
        except: match_time_str, match_time_bj = "时间未知", now_bj
            
        home_team, away_team = match['home_team'], match['away_team']
        try:
            outcomes = match['bookmakers'][0]['markets'][0]['outcomes']
            home_odds = next((item['price'] for item in outcomes if item['name'] == home_team), 0)
            away_odds = next((item['price'] for item in outcomes if item['name'] == away_team), 0)
            draw_odds = next((item['price'] for item in outcomes if item['name'] == 'Draw'), 0)
        except: continue 
        
        ht_elo, at_elo = elo_dict.get(home_team, 1500), elo_dict.get(away_team, 1500)
        features = pd.DataFrame({'home_elo': [ht_elo], 'away_elo': [at_elo], 'elo_diff': [ht_elo - at_elo]})
        probs = model.predict_proba(features)[0]
        
        prob_home = probs[list(classes).index('HomeWin')]
        prob_away = probs[list(classes).index('AwayWin')]
        prob_draw = probs[list(classes).index('Draw')]
        
        ev_home = calculate_ev(prob_home, home_odds) if home_odds else -1
        ev_away = calculate_ev(prob_away, away_odds) if away_odds else -1
        ev_draw = calculate_ev(prob_draw, draw_odds) if draw_odds else -1
        
        best_ev = 0
        best_choice = None
        
        # 核心优化1应用：只有概率 >= min_prob_threshold 且 EV > 0 才会被考虑
        if ev_home > 0 and ev_home > best_ev and prob_home >= min_prob_threshold:
            best_ev, best_choice = ev_home, {'label': f"{home_team} 胜", 'prob': prob_home, 'odds': home_odds}
        if ev_away > 0 and ev_away > best_ev and prob_away >= min_prob_threshold:
            best_ev, best_choice = ev_away, {'label': f"{away_team} 胜", 'prob': prob_away, 'odds': away_odds}
        if ev_draw > 0 and ev_draw > best_ev and prob_draw >= min_prob_threshold:
            best_ev, best_choice = ev_draw, {'label': "平局", 'prob': prob_draw, 'odds': draw_odds}
            
        if best_choice:
            intl_val_matches.append({
                'home_team': home_team, 'away_team': away_team,
                'match_time_obj': match_time_bj, # 用于排序
                'match_time_str': match_time_str,
                'target_label': best_choice['label'],
                'target_odds': best_choice['odds'],
                'prob': best_choice['prob'], 'ev': best_ev,
                'unique_key': f"jc_{home_team}_{away_team}_{best_choice['label']}"
            })

    # 核心优化2：按开球时间先后顺序排列
    intl_val_matches.sort(key=lambda x: x['match_time_obj'])

    if not intl_val_matches:
        st.warning(f"🤷 暂无符合 [胜率 ≥ {min_prob_threshold*100:.0f}%] 且具有正收益的赛事，管住手，放弃下注！")
    else:
        with st.form("jc_odds_form"):
            st.success(f"🔍 截获 {len(intl_val_matches)} 场稳健型价值比赛 (按时间排序)：")
            for vm in intl_val_matches:
                col1, col2, col3, col4 = st.columns([2, 1, 1.5, 2])
                col1.markdown(f"🕒 **{vm['match_time_str']}** <br> {vm['home_team']} vs {vm['away_team']} <br> 🎯 **选: {vm['target_label']}**", unsafe_allow_html=True)
                col2.markdown(f"国际: `{vm['target_odds']}`")
                col3.markdown(f"胜率: `{vm['prob']*100:.1f}%` <br>EV: <span style='color:#2e7d32'>+{vm['ev']*100:.1f}%</span>", unsafe_allow_html=True)
                default_jc = max(1.01, float(vm['target_odds']) - 0.20)
                col4.number_input(f"竞彩【{vm['target_label']}】", min_value=0.00, step=0.01, value=default_jc, key=vm['unique_key'])
                st.divider()
            submitted = st.form_submit_button("⚙️ 生成【Top 5 精选 2串1 方案】", type="primary")

        # ================= 5. 精选 2 串 1 报告与凯利仓位 =================
        valid_jc_matches = []
        for vm in intl_val_matches:
            jc_odds = st.session_state.get(vm['unique_key'], max(1.01, float(vm['target_odds']) - 0.20))
            jc_ev = calculate_ev(vm['prob'], jc_odds)
            if jc_ev > 0:
                valid_jc_matches.append({
                    'match_name': f"({vm['match_time_str']}) {vm['target_label']}",
                    'prob': vm['prob'], 'odds': jc_odds, 'ev': jc_ev
                })

        if len(valid_jc_matches) >= 2:
            st.header("🏆 绝密精算方案 (仅展示 Top 5 最佳 2串1)")
            apply_tax = st.toggle("💸 模拟大额扣税 (超3000扣20%)")
            tax_multi = 0.8 if apply_tax else 1.0
            
            combos_2 = []
            for combo in itertools.combinations(valid_jc_matches, 2):
                c_prob = combo[0]['prob'] * combo[1]['prob']
                c_odds = combo[0]['odds'] * combo[1]['odds']
                eff_odds = c_odds * tax_multi
                c_ev = calculate_ev(c_prob, eff_odds)
                if c_ev > 0:
                    # 计算四分之一凯利仓位
                    k_fraction = calculate_kelly(c_prob, eff_odds, 0.25)
                    combos_2.append({'combo': combo, 'prob': c_prob, 'odds': c_odds, 'eff_odds': eff_odds, 'ev': c_ev, 'kelly': k_fraction})
            
            # 核心优化3：排序并只取前 5
            combos_2.sort(key=lambda x: (x['prob'], x['ev']), reverse=True)
            top_5_combos = combos_2[:5]

            if not top_5_combos:
                st.error("当前条件下无任何盈利 2串1，坚决空仓！")
            else:
                for idx, c in enumerate(top_5_combos):
                    with st.container(border=True):
                        st.subheader(f"🥇 方案 {idx+1}")
                        st.markdown(f"**[1]** {c['combo'][0]['match_name']} <br>**[2]** {c['combo'][1]['match_name']}", unsafe_allow_html=True)
                        
                        col_a, col_b = st.columns(2)
                        col_a.markdown(f"**综合打出概率**: `{c['prob']*100:.1f}%`")
                        if apply_tax:
                            col_a.markdown(f"**税后综合赔率**: `{c['eff_odds']:.2f}`")
                        else:
                            col_a.markdown(f"**综合竞彩赔率**: `{c['odds']:.2f}`")
                            
                        col_b.success(f"📈 预期收益率 (EV): +{c['ev']*100:.2f}%")
                        # 核心优化4：凯利仓位建议
                        rec_percent = c['kelly'] * 100
                        if rec_percent < 0.5:
                            col_b.warning(f"💼 仓位极低: 建议微注娱乐 (占总本金 **0.5%** 以下)")
                        else:
                            col_b.info(f"💼 **严格资金管理**: 建议下注总本金的 **{rec_percent:.2f}%**")
