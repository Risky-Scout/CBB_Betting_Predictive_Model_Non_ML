#!/usr/bin/env python3
"""CBB BETTING SYNDICATE v6.0 - WITH PLAYER BPM"""

import pandas as pd
import numpy as np
import requests
import os
import sys
from datetime import datetime
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
    from player_projections import PlayerProjections, EnhancedInjuryLoader
    PLAYER_PROJ = PlayerProjections()
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

CONFERENCE_HCA = {'B12':4.0,'B10':4.0,'SEC':4.0,'ACC':3.8,'BE':3.8,'MWC':3.8,'WCC':3.5,'A10':3.5,'default':3.5}
ELITE_VENUES = {'Duke':1.2,'Kansas':1.2,'Kentucky':1.0,'Gonzaga':1.0,'Purdue':1.0,'Auburn':1.0}

class BartttorvikLoader:
    def __init__(self):
        self.path = os.path.expanduser("~/cbb_betting/barttorvik_2026.csv")
        self.teams = {}
    def load(self):
        if not os.path.exists(self.path): return False
        df = pd.read_csv(self.path)
        for _,r in df.iterrows():
            n = r['team']
            self.teams[n.lower()] = {'team':n,'rank':int(r['rank']),'conf':r['conf'],'adj_o':float(r['adjoe']),'adj_d':float(r['adjde']),'adj_t':float(r['adjt'])}
        print(f"  ✓ Loaded {len(self.teams)} teams")
        return True
    def get(self,name): return self.teams.get(name.lower().strip())

class OddsAPI:
    def __init__(self,key):
        self.key = key
        self.games = []
    def fetch(self):
        r = requests.get("https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds",params={'apiKey':self.key,'regions':'us','markets':'spreads'},timeout=30)
        if r.status_code != 200: return False
        for g in r.json():
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
                self.games.append({'home':hm,'away':am,'spread':round(np.median(spreads),1)})
        print(f"  ✓ Loaded {len(self.games)} games")
        return True

@dataclass
class Pred:
    home:str; away:str; mkt:float; mdl:float; edge:float
    bet_team:str; bet_line:float; cover:float; kelly:float
    winner:str; margin:float; factors:List[str]=field(default_factory=list)
    @property
    def grade(self):
        if self.edge>=5 and self.cover>=0.65: return "A+"
        if self.edge>=4 and self.cover>=0.58: return "A"
        if self.edge>=3 and self.cover>=0.55: return "B+"
        if self.edge>=2: return "C+"
        return "C"

class Predictor:
    def __init__(self,loader,inj=None,rest=None,players=None):
        self.loader=loader; self.inj=inj; self.rest=rest; self.players=players
        self.enhanced_inj = None
        if inj and players:
            self.enhanced_inj = EnhancedInjuryLoader(inj, players)
    
    def predict(self,home_name,away_name,mkt_spread):
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
        
        if self.enhanced_inj:
            hi,hp=self.enhanced_inj.get_injury_impact(home['team'])
            ai,ap=self.enhanced_inj.get_injury_impact(away['team'])
            adj=(ai-hi)*0.7
            if abs(adj)>0.5:
                h_margin+=adj
                if hi>1: factors.append(f"{home['team']} injuries: -{hi:.1f} pts")
                for p in hp[:2]: factors.append(f"  └ {p}")
                if ai>1: factors.append(f"{away['team']} injuries: -{ai:.1f} pts")
                for p in ap[:2]: factors.append(f"  └ {p}")
        
        if self.rest:
            radj,reason=self.rest.get_rest_adjustment(home['team'],away['team'])
            if abs(radj)>0.3: h_margin+=radj; factors.append(reason) if reason else None
        
        mdl=-h_margin
        winner=home['team'] if h_margin>0 else away['team']
        margin_abs=abs(h_margin)
        edge=abs(mkt_spread-mdl)
        
        if mdl<mkt_spread: bet_team,bet_line=home['team'],mkt_spread
        else: bet_team,bet_line=away['team'],-mkt_spread
        
        exp,mkt=-mdl,-mkt_spread
        cover=1-stats.norm.cdf(mkt,exp,SPREAD_STD) if bet_team==home['team'] else stats.norm.cdf(mkt,exp,SPREAD_STD)
        kelly=max(0,((0.909*cover)-(1-cover))/0.909*KELLY_FRAC) if cover>MIN_COVER else 0
        
        return Pred(home['team'],away['team'],mkt_spread,round(mdl,1),round(edge,1),bet_team,round(bet_line,1),round(cover,4),round(kelly,4),winner,round(margin_abs,1),factors)

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--odds-key',required=True)
    p.add_argument('--bankroll',type=float,default=10000)
    args=p.parse_args()
    global BANKROLL; BANKROLL=args.bankroll
    
    print("\n"+"="*80)
    print("CBB BETTING SYNDICATE v6.0")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%A, %B %d, %Y %I:%M %p')}")
    print(f"Bankroll: ${BANKROLL:,.0f}")
    print("="*80)
    
    print("\n[1] Loading teams...")
    loader=BartttorvikLoader()
    if not loader.load(): return
    
    print("\n[2] Loading player BPM data...")
    if PLAYER_PROJ: PLAYER_PROJ.fetch_all_players()
    
    inj=None
    print("\n[3] Loading injuries...")
    if HAS_INJURIES: inj=InjuryLoader(); inj.load()
    
    rest=None
    print("\n[4] Loading rest data...")
    if HAS_REST: rest=RestTrackerV2(); rest.fetch_recent_games(5)
    
    print("\n[5] Fetching odds...")
    api=OddsAPI(args.odds_key)
    if not api.fetch(): return
    
    print("\n[6] Generating predictions...")
    pred=Predictor(loader,inj,rest,PLAYER_PROJ)
    preds=[pred.predict(g['home'],g['away'],g['spread']) for g in api.games]
    preds=[x for x in preds if x]
    preds.sort(key=lambda x:x.edge,reverse=True)
    
    bets=[x for x in preds if x.edge>=MIN_EDGE and x.cover>=MIN_COVER][:MAX_BETS]
    
    print("\n"+"="*80)
    print("TODAY'S BETS")
    print("="*80)
    
    if not bets:
        print("\n  NO QUALIFYING BETS TODAY")
        return
    
    total=0
    for i,b in enumerate(bets,1):
        amt=min(BANKROLL*b.kelly,BANKROLL*0.03,BANKROLL*0.1-total)
        amt=round(amt/5)*5
        if amt<20: continue
        
        print(f"\n{'-'*80}")
        print(f"BET #{i}: {b.away} @ {b.home}")
        print(f"{'-'*80}")
        print(f"\n  BET:   {b.bet_team} {b.bet_line:+.1f}")
        print(f"  STAKE: ${amt:,.0f}")
        print(f"\n  Market: {b.home if b.mkt<0 else b.away} by {abs(b.mkt):.1f}")
        print(f"  Model:  {b.winner} by {b.margin:.1f}")
        print(f"  Edge:   {b.edge:.1f} pts | Cover: {b.cover:.1%} | Grade: {b.grade}")
        if b.factors:
            print(f"\n  Factors:")
            for f in b.factors: print(f"    • {f}")
        total+=amt
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {len(bets)} bets | ${total:,.0f} stake | {total/BANKROLL*100:.1f}% of bankroll")
    print("="*80)

if __name__=="__main__": main()
