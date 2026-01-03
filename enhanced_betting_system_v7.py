#!/usr/bin/env python3
"""
CBB BETTING SYNDICATE v7.2
==========================
Features:
- Pre-game only (skips live games)
- Player-level BPM injuries
- Minutes redistribution
- Real rebounding data (ORB%, DRB%)
- Pace/style matchup adjustments
- Travel/timezone adjustments
- Rest advantages
- Line movement tracking
- Auto-logging bets
"""

import pandas as pd
import numpy as np
import requests
import os
import sys
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import List
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAPPER = None
try:
    from team_mapping import TeamMapper
    MAPPER = TeamMapper()
except: pass

PLAYER_PROJ = None
try:
    from player_projections import PlayerProjections
    PLAYER_PROJ = PlayerProjections()
except: pass

HAS_MINUTES = False
try:
    from minutes_redistribution import EnhancedInjuryImpact
    HAS_MINUTES = True
except: pass

HAS_MATCHUP = False
try:
    from matchup_adjustments import MatchupAnalyzer
    HAS_MATCHUP = True
except: pass

HAS_TRAVEL = False
try:
    from travel_adjustment import get_travel_adjustment
    HAS_TRAVEL = True
except: pass

HAS_INJURIES = False
try:
    from injury_loader import InjuryLoader
    HAS_INJURIES = True
except: pass

HAS_REST = False
try:
    from data_fetchers import RestTrackerV2
    HAS_REST = True
except: pass

SPREAD_STD = 11.0
KELLY_FRAC = 0.15
MIN_EDGE = 2.0
MIN_COVER = 0.52
MAX_BETS = 5
BANKROLL = 10000.0
MIN_HOURS_UNTIL_GAME = 1.5

CONFERENCE_HCA = {'B12':4.0,'B10':4.0,'SEC':4.0,'ACC':3.8,'BE':3.8,'MWC':3.8,'WCC':3.5,'A10':3.5,'default':3.5}
ELITE_VENUES = {'Duke':1.2,'Kansas':1.2,'Kentucky':1.0,'Gonzaga':1.0,'Purdue':1.0,'Auburn':1.0}

class TeamLoader:
    def __init__(self):
        self.path = os.path.expanduser("~/cbb_betting/barttorvik_2026.csv")
        self.teams = {}
    def load(self):
        if not os.path.exists(self.path): return False
        df = pd.read_csv(self.path)
        for _,r in df.iterrows():
            n = r['team']
            self.teams[n.lower()] = {'team':n,'rank':int(r['rank']),'conf':r['conf'],'adj_o':float(r['adjoe']),'adj_d':float(r['adjde']),'adj_t':float(r['adjt'])}
        print(f"  ✓ {len(self.teams)} teams")
        return True
    def get(self,name): return self.teams.get(name.lower().strip())

class OddsAPI:
    def __init__(self,key):
        self.key = key
        self.games = []
    def fetch(self):
        r = requests.get("https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds",params={'apiKey':self.key,'regions':'us','markets':'spreads'},timeout=30)
        if r.status_code != 200: return False
        
        now = datetime.now(timezone.utc)
        skipped_live = 0
        
        for g in r.json():
            commence = g.get('commence_time', '')
            game_hour_et = 19
            
            if commence:
                try:
                    game_time = datetime.fromisoformat(commence.replace('Z', '+00:00'))
                    hours_until = (game_time - now).total_seconds() / 3600
                    
                    if hours_until < MIN_HOURS_UNTIL_GAME:
                        skipped_live += 1
                        continue
                    
                    game_hour_et = (game_time.hour - 5) % 24
                except:
                    pass
            
            h,a = g.get('home_team',''),g.get('away_team','')
            hm = MAPPER.odds_to_barttorvik(h) if MAPPER else h
            am = MAPPER.odds_to_barttorvik(a) if MAPPER else a
            spreads = []
            for b in g.get('bookmakers',[]):
                for m in b.get('markets',[]):
                    if m['key']=='spreads':
                        for o in m['outcomes']:
                            if o['name']==h: spreads.append(o.get('point',0))
            if spreads and max(spreads)-min(spreads)<=5:
                self.games.append({
                    'home':hm,'away':am,
                    'spread':round(np.median(spreads),1),
                    'game_hour_et': game_hour_et
                })
        
        print(f"  ✓ {len(self.games)} pre-game lines ({skipped_live} live/started skipped)")
        return True

@dataclass
class Pred:
    home:str; away:str; home_rank:int; away_rank:int
    mkt:float; mdl:float; edge:float; bet_team:str; bet_line:float
    cover:float; kelly:float; stake:float; winner:str; margin:float
    factors:List[str]=field(default_factory=list)
    @property
    def grade(self):
        if self.edge>=6 and self.cover>=0.68: return "A+"
        if self.edge>=5 and self.cover>=0.65: return "A"
        if self.edge>=4 and self.cover>=0.60: return "A-"
        if self.edge>=3 and self.cover>=0.55: return "B+"
        if self.edge>=2: return "C+"
        return "C"

class PredictorV7:
    def __init__(self,loader,inj=None,rest=None,players=None,line_tracker=None):
        self.loader=loader; self.inj=inj; self.rest=rest; self.players=players
        self.line_tracker=line_tracker
        self.enhanced_inj = None
        if inj and players and HAS_MINUTES:
            self.enhanced_inj = EnhancedInjuryImpact(inj, players)
        self.matchups = None
        if HAS_MATCHUP:
            self.matchups = MatchupAnalyzer(loader)
    
    def predict(self,home_name,away_name,mkt_spread,game_hour_et=19,bankroll=10000):
        home=self.loader.get(home_name); away=self.loader.get(away_name)
        if not home or not away: return None
        factors=[]
        
        pace=(home['adj_t']*away['adj_t'])/67.5
        h_eff=(home['adj_o']*away['adj_d'])/100
        a_eff=(away['adj_o']*home['adj_d'])/100
        hca=CONFERENCE_HCA.get(home['conf'],3.5)
        h_pts=(h_eff*pace)/100+hca/2
        a_pts=(a_eff*pace)/100-hca/2
        h_margin=h_pts-a_pts
        
        if home['team'] in ELITE_VENUES:
            h_margin+=ELITE_VENUES[home['team']]
            factors.append(f"Elite venue +{ELITE_VENUES[home['team']]}")
        
        if self.enhanced_inj:
            hi,hp=self.enhanced_inj.get_injury_impact(home['team'])
            ai,ap=self.enhanced_inj.get_injury_impact(away['team'])
            adj=(ai-hi)*0.7
            if abs(adj)>0.5:
                h_margin+=adj
                if hi>1: 
                    factors.append(f"{home['team']} injuries: -{hi:.1f}")
                    for p in hp[:2]: factors.append(f"  └ {p}")
                if ai>1: 
                    factors.append(f"{away['team']} injuries: -{ai:.1f}")
                    for p in ap[:2]: factors.append(f"  └ {p}")
        
        if self.rest:
            radj,reason=self.rest.get_rest_adjustment(home['team'],away['team'])
            if abs(radj)>0.3: 
                h_margin+=radj
                if reason: factors.append(reason)
        
        if self.matchups:
            madj,mfactors=self.matchups.get_all_adjustments(home['team'],away['team'])
            if abs(madj)>0.3:
                h_margin+=madj
                for mf in mfactors: factors.append(mf)
        
        if HAS_TRAVEL:
            tadj,treason=get_travel_adjustment(home['team'],away['team'],game_hour_et)
            if abs(tadj)>0.3:
                h_margin+=tadj
                if treason: factors.append(treason)
        
        mdl=-h_margin
        winner=home['team'] if h_margin>0 else away['team']
        margin_abs=abs(h_margin)
        edge=abs(mkt_spread-mdl)
        
        if mdl<mkt_spread: bet_team,bet_line=home['team'],mkt_spread
        else: bet_team,bet_line=away['team'],-mkt_spread
        
        exp,mkt=-mdl,-mkt_spread
        cover=1-stats.norm.cdf(mkt,exp,SPREAD_STD) if bet_team==home['team'] else stats.norm.cdf(mkt,exp,SPREAD_STD)
        
        kelly_mult=1.0
        if self.line_tracker:
            game_key=f"{away_name} @ {home_name}"
            movement=self.line_tracker.get_line_movement(game_key)
            if movement and abs(movement.get('movement',0))>1:
                move=movement['movement']
                your_side='home' if bet_team==home['team'] else 'away'
                if (your_side=='home' and move<-1) or (your_side=='away' and move>1):
                    kelly_mult=1.15; factors.append(f"Sharp $ your way ({move:+.1f}) ↑")
                elif (your_side=='home' and move>1) or (your_side=='away' and move<-1):
                    kelly_mult=0.85; factors.append(f"Line moving against ({move:+.1f}) ↓")
        
        kelly=max(0,((0.909*cover)-(1-cover))/0.909*KELLY_FRAC)*kelly_mult if cover>MIN_COVER else 0
        stake=min(bankroll*kelly,bankroll*0.03,bankroll*0.10)
        stake=round(stake/5)*5
        
        return Pred(home['team'],away['team'],home['rank'],away['rank'],mkt_spread,round(mdl,1),round(edge,1),bet_team,round(bet_line,1),round(cover,4),round(kelly,4),stake,winner,round(margin_abs,1),factors)

def auto_log_bets(bets):
    """Automatically append today's bets to bet_log.csv"""
    log_path = os.path.expanduser("~/cbb_betting/bet_log.csv")
    today = datetime.now().strftime("%Y-%m-%d")
    
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("date,game,bet,line,stake,edge,cover_prob,result,profit\n")
    
    with open(log_path, 'r') as f:
        content = f.read()
        if today in content:
            print(f"\n⚠ Today's bets already in log - skipping auto-log")
            return
    
    with open(log_path, 'a') as f:
        for b in bets:
            game = f"{b.away} @ {b.home}"
            f.write(f"{today},{game},{b.bet_team},{b.bet_line},{b.stake},{b.edge},{b.cover*100:.1f},,\n")
    
    print(f"\n✓ Auto-logged {len(bets)} bets to bet_log.csv")

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--odds-key',required=True)
    p.add_argument('--bankroll',type=float,default=10000)
    p.add_argument('--no-log',action='store_true',help='Skip auto-logging')
    args=p.parse_args()
    global BANKROLL; BANKROLL=args.bankroll
    
    print("\n"+"="*80)
    print("CBB BETTING SYNDICATE v7.2")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}")
    print(f"Bankroll: ${BANKROLL:,.0f}")
    print("="*80)
    
    print("\n[1] Loading teams...")
    loader=TeamLoader()
    if not loader.load(): return
    
    print("\n[2] Loading player BPM...")
    if PLAYER_PROJ: PLAYER_PROJ.fetch_all_players()
    
    print("\n[3] Loading injuries...")
    inj=None
    if HAS_INJURIES: inj=InjuryLoader(); inj.load()
    
    print("\n[4] Loading rest data...")
    rest=None
    if HAS_REST: rest=RestTrackerV2(); rest.fetch_recent_games(5)
    
    print("\n[5] Loading line movement...")
    line_tracker=None
    try:
        from line_movement import LineMovementTracker
        line_tracker=LineMovementTracker(args.odds_key)
        line_tracker.fetch_current_lines()
        print(f"  ✓ Tracking {len(line_tracker.line_history)} games")
    except: print("  ✗ Line tracking unavailable")
    
    print("\n[6] Fetching odds (pre-game only)...")
    api=OddsAPI(args.odds_key)
    if not api.fetch(): return
    
    print("\n[7] Generating predictions...")
    pred=PredictorV7(loader,inj,rest,PLAYER_PROJ,line_tracker)
    preds=[pred.predict(g['home'],g['away'],g['spread'],g.get('game_hour_et',19),BANKROLL) for g in api.games]
    preds=[x for x in preds if x]
    preds.sort(key=lambda x:x.edge,reverse=True)
    bets=[x for x in preds if x.edge>=MIN_EDGE and x.cover>=MIN_COVER and x.stake>=20][:MAX_BETS]
    
    print("\n"+"="*80)
    print("TODAY'S BETS")
    print("="*80)
    
    if not bets:
        print("\n  NO QUALIFYING BETS TODAY")
        print(f"\n  Analyzed: {len(preds)} games")
        print(f"  With 2+ edge: {len([p for p in preds if p.edge >= 2])}")
        return
    
    total=0
    for i,b in enumerate(bets,1):
        print(f"\n{'-'*80}")
        print(f"BET #{i}: {b.away} (#{b.away_rank}) @ {b.home} (#{b.home_rank})")
        print(f"{'-'*80}")
        print(f"\n  BET:   {b.bet_team} {b.bet_line:+.1f}")
        print(f"  STAKE: ${b.stake:,.0f}")
        print(f"  GRADE: {b.grade}")
        print(f"\n  Market: {b.home if b.mkt<0 else b.away} by {abs(b.mkt):.1f}")
        print(f"  Model:  {b.winner} by {b.margin:.1f}")
        print(f"  Edge:   {b.edge:.1f} pts | Cover: {b.cover:.1%}")
        if b.factors:
            print(f"\n  Factors:")
            for f in b.factors: print(f"    • {f}")
        total+=b.stake
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(bets)} bets | ${total:,.0f} stake | {total/BANKROLL*100:.1f}% exposure")
    print(f"Avg Edge: {np.mean([b.edge for b in bets]):.1f} | Avg Cover: {np.mean([b.cover for b in bets]):.1%}")
    print(f"Expected: ${total*0.909*np.mean([b.cover for b in bets])-total*(1-np.mean([b.cover for b in bets])):+,.0f}")
    print("="*80)
    
    print(f"\n{'='*80}")
    print("QUICK REFERENCE")
    print(f"{'='*80}")
    print(f"{'#':<3} {'Game':<32} {'Bet':<18} {'Edge':>5} {'Cover':>6} {'Stake':>7}")
    print(f"{'-'*75}")
    for i,b in enumerate(bets,1):
        game=f"{b.away} @ {b.home}"[:31]
        bet=f"{b.bet_team[:10]} {b.bet_line:+.1f}"
        print(f"{i:<3} {game:<32} {bet:<18} {b.edge:>5.1f} {b.cover:>6.1%} ${b.stake:>6,.0f}")
    print(f"{'-'*75}")
    print(f"{'TOTAL':<56} ${total:>6,.0f}")
    print("="*80)
    
    # Auto-log bets
    if not args.no_log:
        auto_log_bets(bets)

if __name__=="__main__": main()
