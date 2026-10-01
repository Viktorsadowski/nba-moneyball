#!/usr/bin/env python3
"""
Builds report/phi_case_last_ring.pdf: the Philadelphia case on top of the main report (same look).

Numbers in the text are from the September 2026 run. If it gets rerun, the text has to be updated by hand,
the figures come along (report/make_case_figures.py, src/ figures).

  python report/make_case_figures.py
  python report/build_case.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
RFIG = HERE / "figures"
OUT = HERE / "phi_case_last_ring.pdf"

# fonts with the accents we need (Jokić, Dončić, Porziņģis). windows: Times New Roman + Calibri,
# linux: their metric clones Liberation Serif + Carlito
F = "/usr/share/fonts/truetype"
FONTS = {"Serif": [r"C:\Windows\Fonts\times.ttf", f"{F}/liberation/LiberationSerif-Regular.ttf"],
         "Serif-Bold": [r"C:\Windows\Fonts\timesbd.ttf", f"{F}/liberation/LiberationSerif-Bold.ttf"],
         "Sans": [r"C:\Windows\Fonts\calibri.ttf", f"{F}/crosextra/Carlito-Regular.ttf"],
         "Sans-Bold": [r"C:\Windows\Fonts\calibrib.ttf", f"{F}/crosextra/Carlito-Bold.ttf"]}
for name, paths in FONTS.items():
    path = next((p for p in paths if Path(p).exists()), None)
    if path is None:
        raise SystemExit(f"no font found for {name}, tried {paths}")
    pdfmetrics.registerFont(TTFont(name, path))

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
GRID = colors.HexColor("#e6e5e0")
BLUE = colors.HexColor("#2a78d6")
PANEL = colors.HexColor("#f4f3ef")

W, H = A4
MARGIN = 2.1 * cm
TEXT_W = W - 2 * MARGIN

body = ParagraphStyle("body", fontName="Serif", fontSize=10.3, leading=14.2, textColor=INK, alignment=TA_JUSTIFY,
                      spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.2, leading=14, textColor=INK, spaceBefore=9,
                    spaceAfter=4)
title = ParagraphStyle("title", fontName="Sans-Bold", fontSize=24, leading=28, textColor=INK, alignment=TA_LEFT)
subtitle = ParagraphStyle("subtitle", fontName="Sans", fontSize=13, leading=17, textColor=MUTED, spaceBefore=4)
byline = ParagraphStyle("byline", fontName="Sans", fontSize=10, leading=13, textColor=MUTED, spaceBefore=10)
caption = ParagraphStyle("caption", fontName="Sans", fontSize=8.6, leading=11, textColor=MUTED, spaceBefore=3,
                         spaceAfter=10)
abstract = ParagraphStyle("abstract", parent=body, fontSize=10, leading=13.8)
cell = ParagraphStyle("cell", fontName="Sans", fontSize=8.6, leading=10.4, textColor=INK)
cell_head = ParagraphStyle("cell_head", parent=cell, fontName="Sans-Bold", textColor=MUTED)
mono = ParagraphStyle("mono", fontName="Courier", fontSize=8.2, leading=10.4, textColor=INK, leftIndent=8)


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def figure(path, cap, width=TEXT_W):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, P(cap, caption)])


def table(rows, widths, cap=None, align_right_from=1):
    data = [[Paragraph(str(c), cell_head) for c in rows[0]]] + \
           [[Paragraph(str(c), cell) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, GRID),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f6f3")))
    t.setStyle(TableStyle(st))
    parts = [t]
    if cap:
        parts.append(P(cap, caption))
    return KeepTogether(parts)


def box(flow):
    t = Table([[flow]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 10),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def on_page(c, doc):
    c.saveState()
    c.setFont("Sans", 8)
    c.setFillColor(MUTED)
    if doc.page > 1:
        c.drawString(MARGIN, H - 1.25 * cm, "How do we get Bron his last ring?")
        c.drawRightString(W - MARGIN, H - 1.25 * cm, "Viktor Sadowski, September 2026")
    c.drawCentredString(W / 2, 1.2 * cm, str(doc.page))
    c.restoreState()




# ── content ───────────────────────────────────────────────────────────────────

s = []
s += [Spacer(1, 1.2 * cm),
      P("How do we get Bron his last ring?", title),
      P("A trade plan for the 2026-27 Philadelphia 76ers, priced in wins, dollars and picks", subtitle),
      P("Viktor Sadowski · September 2026 · code: nba-moneyball · a case study on top of "
        "\"What is an NBA player worth?\"", byline),
      Spacer(1, 0.6 * cm)]

s.append(box([
    P("Summary", ParagraphStyle("abs_h", parent=h2, spaceBefore=0)),
    P("LeBron James signed with Philadelphia at 41 for the minimum. As the roster stands, our model gives the 76ers "
      "46 wins, a 78% chance to make the playoffs, 10% to reach the Finals and 3.5% to win the title. The question for "
      "the front office: what trades, under which payroll limit, give him the best shot at one more ring, and what "
      "do they cost?", abstract),
    P("Every trade needs two prices. Ours says what a player adds in wins. The other team's says what they will "
      "accept, so we built a second model for that: what the market pays for a player's box score, fitted on 2,635 "
      "contracts, and what the league pays for a pick, backed out of 415 real trades. Over 13 seasons our value "
      "predicts team wins better than both payroll and the market's price, and once ours is known the market's price "
      "adds nothing. That gap is the plan.", abstract),
    P("The market still prices Joel Embiid and Jaylen Brown close to their salaries, while our model has both "
      "about $110M underwater over their contracts. Selling both at the market's price, plus Anfernee Simons, for "
      "Tyrese Haliburton, Bam Adebayo, Aaron Nesmith, Davion Mitchell, Cam Spencer and Cedric Coward takes the 76ers "
      "to 58 wins, a 46% chance of reaching the Finals and 27% of winning it. It stays under the first apron, keeps "
      "15 players with enough bigs and guards to field normal lineups, pays for both players' trade bonuses, cuts the "
      "tax bill from $21M to $13M, and costs all eight second-round picks while keeping the 2033 first. The other "
      "teams come out even by their own measure. No single trade gets close: the three need each other. 60 wins comes "
      "up just short.", abstract),
]))

# 1
s.append(P("1. Where Philadelphia stands", h1))
s.append(P("The 76ers had a busy summer. They traded for Jaylen Brown, signed LeBron James and Kentavious "
           "Caldwell-Pope to minimum deals and kept Tyrese Maxey, Joel Embiid and VJ Edgecombe. Our season simulator "
           "(10,000 seasons with the real schedule format, play-in and best-of-seven series) projects them at 45.9 "
           "wins. The betting market is more optimistic: 50.5 wins at BetMGM and a title price of about 9% on Kalshi "
           "once the prices are scaled to add up to 100%. The difference is mostly Brown and Embiid, and it comes back "
           "in section 4."))
s.append(table([
    ["", "2026-27"],
    ["Payroll (dead money included)", "$213.2M"],
    ["Luxury tax line / first apron / second apron", "$200.4M / $209.0M / $221.7M"],
    ["Position", "$12.8M over the tax, $4.2M over the first apron"],
    ["Tax bill (non-repeater rates)", "$21.2M"],
    ["Tradeable picks", "2033 1st, eight 2nds (the other 1sts are locked by the Stepien rule or owed)"],
    ["Trade exceptions", "$4.2M and $2.3M"],
    ["Projection", "45.9 wins, playoffs 78%, Finals 10%, title 3.5%"],
], [6.2 * cm, 10.6 * cm], "Table 1. The starting point. Payroll from our contract data (Capmath has $213.4M), "
   "pick inventory from PhillyVoice (August 1)."))
s.append(P("The rules of the exercise: trades take effect for game 1 (as if it were July, so the December 15 rule for "
           "new signings is ignored), LeBron stays, 2026 draftees stay where they are, and every trade has to pass the "
           "2023 CBA on both sides. Above the first apron a team can't take back more salary than it sends out, above "
           "the second it can't combine players to match salary. A contender (50%+ playoff odds) won't make itself "
           "worse this season, whatever it gets back."))

# 2
s.append(P("2. What a win is worth to Philadelphia", h1))
s.append(figure(RFIG / "case_curve.png", "Figure 1. Philadelphia's odds by projected wins, everything else fixed. "
                "The Finals and title curves are steep exactly where the plan lands."))
s.append(P("Moving the 76ers up the curve shows the two targets. 52 wins means a 95% chance of the playoffs, about "
           "3 points per 100 possessions better than now. 60 wins means about a 50% chance of the Finals, "
           "roughly 7.5 points better. Wins are not worth the same everywhere: going from 46 to 50 wins adds 13 points "
           "of playoff odds but only 4 of title odds, from 55 to 60 it adds 16 points of title odds. For a team "
           "playing for a ring, the last wins are the valuable ones."))

# 3
s.append(P("3. Two prices for every player", h1))
s.append(P("Our value is the main model: RAPM with a box-score prior, aged, injury-adjusted and turned into wins "
           "above replacement at $8.4M a win. It is what a player adds. It is not what another general manager will "
           "accept for him, and a trade needs both."))
s.append(P("3.1 The market's price", h2))
s.append(P("So the second model prices players the way the market does. It takes every new veteran contract since "
           "2013-14 (2,635 of them, found where a salary jumps outside the raises one deal allows) and fits the "
           "first-year salary as a share of the cap on what the player had done before he signed: points, rebounds, "
           "assists, minutes and games, last season and the three before, plus shooting, age and our RAPM value. "
           "Minimum and max contracts only say \"at most\" and \"at least\", so the fit is a Tobit. Rookie-scale deals "
           "are left out, the CBA sets those."))
s.append(figure(FIG / "market_value.png", "Figure 2. New veteran contracts: predicted with the season left out vs "
                "what was signed. Error 3.1% of the cap on deals between the minimum and the max, about $5M a year.",
                width=TEXT_W * 0.72))
s.append(P("Three things stand out. Consistency is paid for: three seasons of stats predict better than the last one "
           "(error 4.4% of the cap vs 4.7%), and games played over three years is one of the strongest terms. The "
           "market pays for volume first. And it pays a little for impact: adding our RAPM value brings the error to "
           "4.3%, but RAPM alone does about as well as the box score. A player's market value is then his market "
           "price for every season left on his contract minus his salary, the same way our surplus works."))
s.append(figure(RFIG / "case_two_prices.png", "Figure 3. Every player under contract: the market's value against "
                "ours, both over the whole contract. Philadelphia's moves in color. Embiid and Brown are where the "
                "two disagree most."))

# 4
s.append(P("4. Who is right?", h1))
s.append(P("The plan below is a bet that our model is right about Embiid and Brown and the market is wrong. That can be "
           "tested on history with the one judge neither side controls: wins. For every team-season since 2013-14 its "
           "players were valued three ways before the season (what the team paid them, what the market model says "
           "their stats were worth, and our projected WAR), each scaled by how much they then played. Which one "
           "predicts the team's wins?"))
s.append(figure(FIG / "arbitrage.png", "Figure 4. Error predicting team wins per 82 games, 390 team-seasons, each "
                "season predicted from the others.", width=TEXT_W * 0.8))
s.append(P("Payroll misses by 8.0 wins, the market model by 7.2 and our model by 6.4. Put all three in one "
           "regression and ours carries it: +8.9 wins per standard deviation, the market model −0.7 (95% interval "
           "−2.7 to +1.5) and payroll +1.7 (+0.4 to +3.1, probably what teams know about health that the stats don't). "
           "Once our value is known, the market's price tells you nothing more about wins. Player by player it points "
           "the same way: among the 10% biggest disagreements, the next two seasons ended up closer to our number in "
           "57 to 64% of cases, though that check uses realized RAPM and so leans our way."))
s.append(P("So the market is a good guide to what other teams will accept, and a worse guide to what a player is worth "
           "on the court. That is the whole opportunity."))

# 5
s.append(P("5. What a pick is worth", h1))
s.append(P("Picks are the other currency, and there are two ways to price them. What a pick turns into: the market "
           "value of what drafted players did over their four rookie-scale seasons, minus their salaries, by slot. "
           "And what trades pay: if players are priced like the market prices them, then in every trade the gap "
           "between the two sides' players is what the picks and cash had to make up. 415 trades since 2013-14 give "
           "one equation per team, solved by least squares with a bootstrap over trades."))
s.append(figure(RFIG / "case_picks.png", "Figure 5. What picks turned into by slot (bars, smoothed in blue) vs "
                "what trades pay for a future first and a second (orange)."))
s.append(table([
    ["", "Future 1st", "2nd"],
    ["What it turns into (average slot)", "$17.9M", "$3.3M"],
    ["What trades pay", "$5.3M (95%: −$0.4M to $11.3M)", "$1.4M ($0.2M to $2.9M)"],
    ["In wins, at $8.4M a win", "0.6", "0.2"],
], [6.0 * cm, 6.0 * cm, 4.8 * cm], "Table 2. Picks at the 2026-27 cap. The slot curve has the same shape as the "
   "published research: #10 is worth about half of #1 and #30 a fifth (Kubatko's four-year curve: 45% and 19%)."))
s.append(P("The trade price is noisy, but two things hold. Trades add up far better in the market's money than in "
           "ours (average miss 5.5% of the cap vs 9.7%), so other teams really do trade on box-score value. And a "
           "traded pick buys less than it turns into on average. For Philadelphia, with one tradeable first and eight "
           "seconds, that means picks can close small gaps but can't buy a star."))

# 6
s.append(P("6. The playoffs", h1))
s.append(P("Titles are won in the playoffs, and rotations get shorter there. From playoff play-by-play since "
           "2010-11 (251 playoff teams), a team's top five players go from 57% of the minutes in the regular season "
           "to 61%, and players ranked 9th and below from 18% to 16%. That should make depth worth less and stars more."))
s.append(figure(FIG / "playoff_rotation.png", "Figure 6. Minutes per game by a player's rank on his team, regular "
                "season vs playoffs.", width=TEXT_W * 0.72))
s.append(P("It is real but small. Team strength with playoff minute shares predicts playoff games only a little "
           "better than with regular-season shares (error 13.90 vs 13.91 points per game). In the simulator it moves "
           "teams between +0.2 and +0.8 points per 100 in the playoffs, most for star-led Denver and Oklahoma City. "
           "Every result below uses it. For a roster built on depth it costs about 1.5 points of title odds."))
s.append(P("6.1 Positions and roster spots", h2))
s.append(P("Everything so far adds player values up, so five centers would count the same as a normal lineup. To check "
           "that, every player gets a role from his box score, not his listed position: rebounds, blocks, assists and "
           "three-point attempts per 36 minutes, clustered into guards, wings and bigs. Then for 467,485 stints since "
           "2013-14 the margin was compared with what the ten players' RAPM says it should be, by how many bigs and "
           "guards each side had on the floor."))
s.append(table([
    ["Lineup", "Share of possessions", "Points per 100 vs 2 bigs / 2 guards (95%)"],
    ["0 bigs", "11%", "−0.8 (−1.4 to −0.1)"],
    ["1 big", "59%", "0.0 (−0.4 to +0.5)"],
    ["3+ bigs", "2%", "−0.5 (−1.6 to +1.1)"],
    ["0 guards", "10%", "−2.1 (−2.7 to −1.5)"],
    ["1 guard", "56%", "−0.7 (−1.1 to −0.3)"],
], [3.6 * cm, 4.0 * cm, 9.2 * cm], "Table 3. Lineup shape on top of the players' values. One big is as good as "
   "two, none costs a little. Ball handlers matter more."))
s.append(P("So position does matter, mostly at the guard spots. How often a team ends up in those lineups follows from "
           "how many bigs and guards play its minutes (fitted over 390 team-seasons), and that now goes into team "
           "strength. Today it ranges from −0.8 points per 100 (Houston, short on guards) to +0.6 (Milwaukee). The "
           "search also counts roster spots: after every trade a team may have at most 15 guaranteed contracts, a "
           "team pushed over waives its cheapest guaranteed player (the money stays, and the partner's loss has to be "
           "covered in the trade), and Philadelphia needs at least 14 standard contracts, topped up with minimum "
           "signings if needed."))

# 7
s.append(P("7. The plan", h1))
s.append(P("The search tries up to three trades in a row. Philadelphia sends one to three players plus picks and gets "
           "zero to two back. Every trade has to be legal for both teams and fair to the partner at the market's "
           "price, with picks at what trades pay for them. Philadelphia adds seconds first, its first only if needed. "
           "Among thousands of legal paths it keeps the strongest rosters, then the full simulator (with the partners "
           "changed too) picks the one with the best title odds. It does that under three payroll limits, since the "
           "owner decides how far past the tax he goes."))
s.append(table([
    ["Scenario", "Wins", "Playoffs", "Finals", "Title", "Payroll", "Tax", "Picks out"],
    ["No trades", "45.9", "78%", "10%", "3.5%", "$213.2M", "$21.2M", ""],
    ["Best under the tax line", "54.7", "98%", "34%", "17.1%", "$200.4M", "$0", "2033 1st + 8 2nds"],
    ["One trade only, 1st apron", "49.2", "88%", "17%", "6.6%", "$208.8M", "$13.1M", "2 2nds"],
    ["One trade only, 2nd apron", "51.4", "94%", "23%", "10.1%", "$212.5M", "$19.6M", "2033 1st + 6 2nds"],
    ["Best under the 1st apron", "58.3", "100%", "46%", "26.8%", "$208.7M", "$13.0M", "8 2nds"],
    ["Best under the 2nd apron", "58.4", "100%", "48%", "28.1%", "$209.5M", "$14.4M", "2033 1st + 8 2nds"],
], [4.4 * cm, 1.2 * cm, 1.6 * cm, 1.4 * cm, 1.3 * cm, 1.9 * cm, 1.7 * cm, 3.3 * cm],
    "Table 4. Scenarios, full simulation with 10,000 seasons, trade bonuses included. The second apron buys 1.3 "
    "points of title odds for the 2033 first and $1.4M more tax, the first apron plan is the one to make."))
s.append(figure(RFIG / "case_scenarios.png", "Figure 7. Odds per scenario."))
s.append(P("7.1 The recommended package", h2))
s.append(table([
    ["Trade", "Philadelphia sends", "Philadelphia gets", "Salary out / in", "Partner's gain (market)",
     "Our gain (model)"],
    ["1. Memphis", "Anfernee Simons, Ariel Hukporti, Dominick Barlow + 2 2nds", "Cam Spencer, Cedric Coward",
     "$12.8M / $8.4M", "+$1.0M", "+$119M"],
    ["2. Indiana", "Jaylen Brown, Justin Edwards, Tacko Fall + 3 2nds", "Tyrese Haliburton, Aaron Nesmith",
     "$60.8M / $59.9M", "+$0.5M", "+$108M"],
    ["3. Miami", "Joel Embiid, Kentavious Caldwell-Pope, Jameer Nelson Jr. + 3 2nds", "Bam Adebayo, Davion Mitchell",
     "$61.9M / $62.2M", "+$0.5M", "+$85M"],
], [1.7 * cm, 4.2 * cm, 3.4 * cm, 2.4 * cm, 2.5 * cm, 2.6 * cm],
    "Table 5. Best under the first apron, in this order. Each partner comes out slightly ahead by the market's "
    "measure, after Memphis waives one guaranteed contract to stay at 15 and Indiana pays Haliburton's trade bonus "
    "($6.9M). Our "
    "gain is value over the contracts, after picks and Brown's trade bonus ($5.5M, paid by Philadelphia)."))
s.append(P("The logic is the same in all three. Brown and Embiid go at the market's price, which is close to even, "
           "while our model sees each of them costing about $110M more than he's worth over his deal. Simons is the "
           "opposite case: the market likes a 28-year-old who scored 18 a game, our model has him at replacement level. "
           "Coming back are players our model rates above their price: Haliburton (4.0 points per 100, coming back from "
           "his Achilles), Adebayo (3.2), Coward and Mitchell (2.2 and 2.1), Spencer and Nesmith (1.8 and 1.7). Almost "
           "4,000 minutes that went to players at or below replacement level (Simons, Edwards, Hukporti and the "
           "camp contracts) now go to real rotation players, and that's where most of the 12 wins come from. The shape "
           "works too: on average one big and two guards on the floor, with Adem Bona and Jabari Walker behind "
           "Adebayo."))
s.append(P("The cost: all 8 second-round picks (the 2033 first stays), $5.5M in cash for Brown's trade bonus, and "
           "$9M by the market's measure, most of it the picks that pay for the two trade bonuses. In return the "
           "payroll drops $4.5M and the tax $8M this season. By our model the 76ers gain about "
           "$310M of value over the "
           "contracts, because the two biggest contracts leave. LeBron himself ends up the 10th most valuable player "
           "on the roster, at 42 he's a rotation player now. If he gets his ring, it will be because of the people "
           "around him."))
s.append(P("7.2 The two targets, and one trade at a time", h2))
s.append(P("52 wins is reached on the way. Even the cheapest route lands at 56.5 wins (Brown to Indiana, Simons to "
           "Memphis, Embiid with Caldwell-Pope to New Orleans for Zion Williamson and Herbert Jones, four second-round "
           "picks), and under the tax line the 76ers can get to 55 wins without paying any tax."))
s.append(P("Not one trade at a time. The best single deal under the first apron is the Memphis one: 49 wins and 6.6% "
           "title odds. Allow the second apron and it's Embiid to Miami for Adebayo and Mitchell: 51 wins and 10%. The "
           "three together get to 27%, over three times what the best one does alone, because title odds climb faster as a "
           "team gets better (Figure 1). For an owner that means approving the package, not a first step."))
s.append(P("60 wins comes up just short. The best roster any trade path finds is 58.4 wins with a 48% chance of the "
           "Finals, against the 50% the target asked for. The limit isn't picks. Stars cost $50M+ in salary, "
           "Philadelphia's big salaries are the two contracts the market won't pay a premium for, and the teams with "
           "stars to spare are contenders that won't get weaker."))

s.append(P("7.3 The fine print", h2))
s.append(P("Three contract details can kill a trade that works on paper. Trade bonuses: Brown has one of 7% (at most "
           "$7M) and Haliburton one of 15%, paid in cash by the team that trades them and added to the salary of the "
           "team that gets them. Both are paid close to the maximum, and a bonus can't lift a salary past it, so this "
           "season only $0.7M and $0.6M land on the cap ($5.5M and $6.9M in total). Small, but in the first version "
           "of the plan Brown's alone pushed Indiana $76,000 over the first apron, where it can't take back more "
           "salary than it sends, and the trade broke. The search now counts every bonus. No-trade clauses: none of "
           "the nine players in the plan can veto a trade. Base-year compensation now only exists for sign-and-trades, "
           "so it doesn't apply here."))

# 8
s.append(P("8. What could go wrong", h1))
s += bullets([
    "The whole plan is a bet against the market. The market has Philadelphia 5 wins higher than we do today, mostly "
    "because it likes Brown and Embiid more. Section 4 says our model has been the better bet historically, but "
    "for any two players it can be wrong.",
    "Haliburton missed all of 2025-26 with a torn Achilles. We expect 9 more games missed and a slower first 20 back, "
    "and he is the most valuable player coming in. The first thing to check before any deal.",
    "\"The partner accepts\" means fair at market prices, and the market model misses by about $5M a year. It is not a "
    "negotiation, and Indiana trading Haliburton or Miami taking Embiid's contract is not a given.",
    "Positions come in as box-score roles and the lineup-shape effect of section 6.1. That catches a roster "
    "without bigs or ball handlers, not finer fit like spacing or two players who need the ball. And a player "
    "keeps his old team's minutes, Spencer or Coward may play more or less in Philadelphia.",
    "Trade bonuses are in at the worst case: Brown's counted again after his July trade, nobody waives one.",

    "The search covers thousands of trade paths, not all (no four-team trades). Tax at non-repeater rates.",

])

# appendix
s.append(PageBreak())
s.append(P("Appendix A. Method notes", h1))
s.append(table([
    ["Piece", "How", "Check"],
    ["Season simulator", "net per 100 from rosters (best players first, filler at replacement), game margin = "
     "strength gap + home court 1.94 + noise sd 13.5, team strength sd 2.56 around the projection, real schedule "
     "format, play-in, 2-2-1-1-1 series", "wins add up to 1,230; team wins MAE 6.1"],
    ["Rookies and young players", "draft slot -> rookie value and minutes (log pick), bump for seasons 2-4 by pick "
     "group learned from earlier seasons", "value backtest RMSE 1.546 -> 1.527, seasons 1-4 1.642 -> 1.585"],
    ["Rosters", "nba.com rosters for 2026-27; a contract row whose player isn't on that team = dead money (payroll, "
     "not the court)", "24 dead-money rows, $138M"],
    ["Current injuries", "remaining games from earlier absences of the same type that lasted as long; for players on "
     "a roster the never-ended cases are left out", "Haliburton 27 -> 9 games"],
    ["Market value", "Tobit on first-year salary share of new veteran contracts (censored at 3% and 24.5% of the "
     "cap), box stats last season + 3 seasons, age, RAPM value", "season left out: MAE 3.1% of cap, r 0.71"],
    ["Pick prices", "value in - out = p1 (1sts out - in) + p2 (2nds out - in) + cash + swaps, least squares, "
     "bootstrap over trades; trades with players we can't price left out", "415 trades, 2013-14 to 2026-27"],
    ["Playoff rotations", "minute share by rank on the team, playoffs vs regular season; team strength with each; "
     "the difference is added in the playoff rounds", "playoff game RMSE 13.913 -> 13.901"],
    ["Positions", "roles from box stats per 36 (rebounds, blocks, assists, 3PA), k-means into guard / wing / big; "
     "stint margin minus the ten players' RAPM ~ number of bigs and guards on each side; team shares of those "
     "lineups from its average number on the floor", "467,485 stints; bootstrap over games"],
    ["Trade bonuses", "% of the remaining contract (or the flat cap), spread like the salary, each season capped by "
     "the player's max (25/30/35% of the cap); paid in cash by the team that trades him, this season's part on the "
     "cap of the team that gets him", "35 of 38 listed kickers matched (ShamSports)"],
    ["Roster spots", "after every trade at most 15 guaranteed contracts (over = waive the cheapest, money stays, "
     "the partner's loss counted), at least 14 standard for PHI (minimum signings)", "all plans in Table 4 pass"],
    ["Trade search", "2-team trades, up to 3 in a row, 1-3 players out + picks, 0-2 in; CBA matching by apron for "
     "both teams; partner even or better at market prices; contenders don't get weaker; beam width 8",
     "full simulation for the finalists"],
    ["Who is right", "team wins ~ payroll / market price / our WAR of the roster before the season, each scaled "
     "by minutes played vs expected", "leave one season out, 390 team-seasons"],
], [3.0 * cm, 9.2 * cm, 4.6 * cm]))
s.append(P("Appendix B. Reproducing it", h1))
s.append(P("After the main pipeline (see the README), in order. The nba.com and basketball-reference pulls run from "
           "a normal machine:"))
for line in ["draft, rookies, value, war, surplus, season_sim", "rosters, scrape_salaries --contracts-only, current_injuries",
             "market_value, scrape_transactions, trade_value", "playoff_stints, playoff_rotation, positions, season_sim, arbitrage",
             "trade_search (about 30 minutes)", "report/make_case_figures.py, report/build_case.py"]:
    s.append(P(line, mono))
s.append(Spacer(1, 6))
s.append(P("Data as in the main report, plus nba.com rosters (nba_api), basketball-reference transactions and "
           "contracts, BetMGM win totals (Yahoo Sports, July 28) and Kalshi title prices (September 24) for the "
           "comparison, cap numbers from the NBA's 2026-27 announcement, the 76ers' pick inventory from PhillyVoice, "
           "the draft pick research summary by Tony ElHabr, trade bonuses from ShamSports and veto rights from Hoops "
           "Rumors (September 2026).", caption))

# headings stick to whatever comes right after them (keepWithNext alone didn't do it with KeepTogether blocks)
def flat(*fs):
    # nested KeepTogether makes reportlab push the block to a new page, so unpack them
    out = []
    for f in fs:
        out += f._content if isinstance(f, KeepTogether) else [f]
    return out


story, i = [], 0
while i < len(s):
    f = s[i]
    if isinstance(f, Paragraph) and f.style.name in ("h1", "h2") and i + 1 < len(s):
        nxt = s[i + 1]
        # h1 directly followed by h2: take the paragraph after the h2 along too
        if isinstance(nxt, Paragraph) and nxt.style.name == "h2" and i + 2 < len(s):
            story.append(KeepTogether(flat(f, nxt, s[i + 2])))
            i += 3
            continue
        story.append(KeepTogether(flat(f, nxt)))
        i += 2
        continue
    story.append(f)
    i += 1
s = story

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.0 * cm,
                        bottomMargin=2.0 * cm, title="How do we get Bron his last ring?", author="Viktor Sadowski")
doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
print(f"report -> {OUT}")
