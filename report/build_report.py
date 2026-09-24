#!/usr/bin/env python3
"""
Builds report/nba_moneyball_report.pdf from the text below + the figures.

Numbers in the text are from the September 2026 run (seasons 2010-11 to 2025-26). If the pipeline gets
rerun on new data, the text has to be updated by hand, the figures come along automatically.

  python report/make_figures.py
  python report/build_report.py
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
OUT = HERE / "nba_moneyball_report.pdf"

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
        c.drawString(MARGIN, H - 1.25 * cm, "What is an NBA player worth?")
        c.drawRightString(W - MARGIN, H - 1.25 * cm, "Viktor Sadowski, September 2026")
    c.drawCentredString(W / 2, 1.2 * cm, str(doc.page))
    c.restoreState()


# ── content ───────────────────────────────────────────────────────────────────

s = []
s += [Spacer(1, 1.2 * cm),
      P("What is an NBA player worth?", title),
      P("Pricing wins, aging and injury risk with free data, 2010-11 to 2025-26", subtitle),
      P("Viktor Sadowski · September 2026 · code: nba-moneyball", byline),
      Spacer(1, 0.6 * cm)]

s.append(box([
    P("Summary", ParagraphStyle("abs_h", parent=h2, spaceBefore=0)),
    P("This paper prices every NBA player in wins and compares that with what he is paid. It uses only free public "
      "data: 16 seasons of play-by-play split into 577,807 lineup stints, two injury sources and salary and contract "
      "data. Player impact comes from regularized adjusted plus-minus (RAPM). It is projected forward with an aging "
      "curve and an injury-risk model, turned into wins above replacement (WAR) by calibrating on team results, and "
      "priced at what the market paid per win: about $13.5M in 2025-26. The model predicts team wins from the "
      "previous season's player values with a correlation of 0.78.", abstract),
    P("Five findings matter for a front office. Players peak at 27 and offense fades faster than defense. Injury "
      "proneness is real and predictable, but players who come back are as good as before, with one exception: after "
      "an ACL tear the first 20 games back are 0.4 to 0.8 points per 100 possessions worse. The biggest surpluses sit "
      "on young players on rookie or early extension deals, while only 10 of the 27 players paid $45M or more "
      "project to earn their salary next season. Three-point volume went from underpaid to overpaid around 2019-20, "
      "but what the market really overpays is scoring. And defense does not win championships: in the playoffs a "
      "point of defensive edge is worth no more than a point of offensive edge.", abstract),
]))

# 1
s.append(P("1. The idea", h1))
s.append(P("A team has a budget, the salary cap, and wants wins. So the useful question about a player is how many wins "
           "he adds per dollar, compared with what else the same money could buy. Baseball has priced players this way "
           "for decades with WAR. In basketball the pieces exist, plus-minus models on one side and public contract "
           "data on the other, but they are rarely put together into one price. This project does that, end to end, "
           "with data anyone can download."))
s.append(P("The framework answers four questions:"))
s += bullets(["How many wins will a player add next season, and over the rest of his contract?",
              "How much of that is at risk because of age and injuries?",
              "What does a win cost on the market?",
              "Where does the market misprice players?"])
s.append(P("The goal is a blueprint: a pipeline a front office can rerun every summer, with clear inputs, a validation "
           "for each step and known limits. Three side questions run on the same data: do MVP voters see the same "
           "players as the model, are three-point shooters overpaid, and does defense win championships."))

# 2
s.append(P("2. Data", h1))
s.append(table([
    ["Source", "Content", "Seasons"],
    ["Play-by-play, stats.nba.com format (free mirror on GitHub)", "every event incl. substitutions, regular season "
     "and playoffs", "2010-11 to 2025-26"],
    ["stats.nba.com via nba_api", "game logs, minutes, ages, box scores, schedules", "2010-11 to 2025-26"],
    ["ProSportsTransactions injury log (pre-scraped CSV)", "injury list moves with a note on the injury",
     "2010-11 to 2019-20"],
    ["Official NBA injury reports (PDF)", "status and reason per player, every game day", "2021-22 to 2025-26"],
    ["Basketball-Reference", "salary per player-season, current contracts, MVP votes", "2010-11 to 2031-32"],
], [5.6 * cm, 7.4 * cm, 3.8 * cm], "Table 1. Data sources. 2020-21 has no injury data and is skipped wherever an "
   "injury count is needed."))
s.append(P("Lineups are rebuilt from the substitutions. Every game becomes a sequence of stints, stretches where the "
           "same ten players are on the floor, with points and possessions for both sides. Between 97.9% and 99.9% of "
           "each season's games pass the consistency checks. That gives 18,996 regular-season games and 1,339 playoff "
           "games. Salaries were matched to player ids by name and team, contracts by name over the last three "
           "seasons."))

# 3
s.append(P("3. Method", h1))
s.append(figure(RFIG / "pipeline.png", "Figure 1. The pipeline. Blue: how good a player is. Orange: how often he "
                "plays. Grey: what he costs. Every box is one script in the code."))
s.append(P("3.1 Impact per possession (RAPM)", h2))
s.append(P("For each season a ridge regression explains points per 100 possessions in every stint with the ten players "
           "on the floor, offense and defense as separate columns, weighted by possessions. The penalty is chosen by "
           "cross-validation with whole games held out (1,000 to 3,000). The result is O-RAPM and D-RAPM: points per "
           "100 possessions a player adds on offense and saves on defense, compared with an average player."))
s.append(P("3.2 Projection and aging", h2))
s.append(P("A single season of RAPM is noisy, so next season's value is a blend of the last three: weights 50/30/20, "
           "each multiplied by minutes played, plus 1,000 minutes at league average to pull small samples towards the "
           "middle. Each past season is first moved along the aging curve to the age the player will be next season. "
           "The aging curve is a fixed-effects regression (player talent plus one term per age) on RAPM, checked "
           "against the delta method and smoothed with a quadratic. In a backtest the age adjustment lowers the error "
           "of the next-season projection from 1.903 to 1.888 points per 100 (RMSE, minutes weighted)."))
s.append(P("3.3 From points to wins", h2))
s.append(P("The blended value is shrunk, so it is put back on a real scale by calibrating against the next season's "
           "team point differentials: calibrated value = 1.70 × value − 0.54. Replacement level is what teams actually "
           "got from near-minimum contracts (the bottom 20% of salaries): −0.61 points per 100. One win equals 33.6 "
           "points of season point differential. Then"))
s[-1] = KeepTogether([s[-1], P("WAR = (value − replacement) × possessions / 100 / 33.6", mono)])
s.append(P("As a check, each team's wins are predicted from the previous season's player values and this season's "
           "minutes: correlation 0.78, mean absolute error 6.2 wins per 82 games. A team of replacement players wins "
           "about 34 games."))
s.append(P("3.4 Availability and injury risk", h2))
s.append(P("Games lost to injury come from the injury log up to 2019-20 and from the official reports after that. The "
           "two sources differ in how many short absences they catch, so injury games are normalized within each "
           "season and converted back with the official-report average (13.7 games per 82). A linear model on the "
           "last three seasons of injuries, age and minutes predicts next season's games missed. In a rolling backtest "
           "it beat both the league average and gradient boosting (mean absolute error 12.4, 12.9 and 12.5 games). "
           "Projected minutes are expected games times minutes per game."))
s.append(P("3.5 The price of a win, and surplus", h2))
s.append(P("The market price of a win in a season is the total salary paid above the near-minimum level, divided by "
           "the total positive projected WAR it bought. It was $13.5M in 2025-26 and grows with league payroll, 9.1% a "
           "year over the last ten seasons. Later contract years move the player further along the aging curve with "
           "the same availability, and WAR is floored at 0 since a team can always bench a player. Surplus over a "
           "contract is the sum over its years of WAR × price per win − salary."))

# 4
s.append(P("4. Results", h1))
s.append(P("4.1 Who is valuable", h2))
s.append(table([
    ["Player", "Age", "Value per 100", "Expected injury games", "WAR", "WAR if healthy", "Lost to injury risk"],
    ["Shai Gilgeous-Alexander", "28", "7.7", "12.1", "11.0", "13.1", "2.1"],
    ["Nikola Jokić", "32", "6.2", "13.1", "8.8", "10.8", "1.9"],
    ["Victor Wembanyama", "23", "5.3", "14.4", "7.1", "8.6", "1.6"],
    ["Derrick White", "32", "4.4", "11.6", "6.4", "7.6", "1.2"],
    ["Amen Thompson", "24", "3.8", "11.1", "6.0", "7.0", "1.0"],
    ["Chet Holmgren", "25", "4.6", "14.5", "5.9", "7.2", "1.3"],
    ["Donovan Mitchell", "30", "3.9", "13.4", "5.6", "6.8", "1.2"],
    ["Luka Dončić", "28", "3.5", "14.7", "5.5", "6.9", "1.3"],
    ["Kawhi Leonard", "36", "4.4", "17.2", "5.4", "7.1", "1.7"],
    ["Giannis Antetokounmpo", "32", "4.1", "17.7", "5.3", "7.0", "1.7"],
], [4.6 * cm, 1.1 * cm, 2.2 * cm, 2.8 * cm, 1.4 * cm, 2.3 * cm, 2.4 * cm],
    "Table 2. Highest projected WAR for 2026-27. Value is the calibrated, age-adjusted value in points per 100 "
    "possessions above average."))
s.append(P("The model and the MVP voters mostly agree on who the best players are. The median MVP winner since 2010-11 "
           "ranks second in that season's RAPM, and 75% of winners were in the RAPM top five. They disagree on why: "
           "vote share correlates with offensive RAPM (Spearman 0.42) and not at all with defensive RAPM (0.04). "
           "Voters reward what they can see on the box score."))

s.append(P("4.2 Aging", h2))
s.append(figure(FIG / "aging_curve.png", "Figure 2. Aging curve, points per 100 possessions compared with the peak. "
                "Fixed effects (lines, smoothed) and the delta method (dashed) agree up to about 33."))
s.append(P("Players peak at 27 and stay close to the peak from 26 to 29. A player is half a point per 100 below his "
           "peak at 23 and again at 32, and 1.4 points below at 20 and at 35. Offense drives both the rise and the "
           "fall; defense moves about a third as much. In money: a four-year deal signed at 30 covers exactly the "
           "years where the curve gets steep."))

s.append(P("4.3 Injuries", h2))
s.append(figure(FIG / "injury_proneness.png", "Figure 3. Left: games missed to injury next season, by injury history "
                "over the last three seasons. Right: next season RAPM compared with the projection, by the same "
                "quintiles. 95% bootstrap intervals."))
s.append(P("Injury proneness is real. Games lost to injury correlate from one season to the next (Spearman 0.33), and "
           "16% of the variation in injury rate belongs to the player. Players in the bottom fifth of injury history "
           "miss 7.0 games the next season, the top fifth 21.1. What injury history does not predict is decline: next "
           "season's RAPM compared with the projection is flat across all five groups. The discount for an "
           "injury-prone player belongs on his expected games, and the talent estimate can stay as it is."))
s.append(P("A season is a clumsy unit for the question of what an injury does. A player hurt in October is back by "
           "January and his next season is really his second season back, while a player hurt in March starts the "
           "next season straight out of rehab. So each absence of 10 or more games is also measured in games played "
           "around it: his last 82 games before the injury as the baseline, and his first 20 and first 82 games after "
           "he is back. Impact in each window comes from one ridge over all 16 seasons, with the post-injury windows "
           "fitted as a change from the same player's baseline. The same windows around random points in healthy "
           "stretches serve as the control group. That gives 1,295 injuries and 2,544 control stretches."))
s.append(figure(FIG / "injury_windows.png", "Figure 4. Injuries measured in games. Left: share who play 41+ games "
                "within two years, compared with the controls. Middle and right: RAPM change from before the injury, "
                "compared with the controls, over the first 20 and the first 82 games back. Blue is the estimate after "
                "empirical-Bayes shrinkage towards the average type, which is the one to believe for small groups."))
s.append(P("Getting back on the floor depends on how long the absence was: 92% of the controls play 41 games within two "
           "years, 7 points fewer after 10-19 games out and 58 points fewer after a full season out. Once the noise is "
           "removed, injury types differ by only about 4 points. Knee injuries other than the ACL are the worst "
           "group (−17). The ACL is the one injury with clear rust: the first 20 games back are 0.78 points per 100 "
           "worse (interval −1.29 to −0.25, 0.41 after shrinkage), and over the first 82 games most of it is gone. "
           "Other types stay within about ±0.2 points. The one exception, +0.27 for 22 Achilles returners, mostly reflects "
           "the aging correction for older players who sat out a full year."))
s.append(P("In the projections the injury risk costs about 20% of the league's healthy WAR. For the 24 players worth 4+ "
           "WAR when healthy it is 1.1 WAR a season on average, 18% of their value, and above 30% for Joel Embiid and "
           "Kyrie Irving."))

s.append(P("4.4 Surplus: who is worth his contract", h2))
s.append(figure(RFIG / "market.png", "Figure 5. Projected WAR against salary for 2026-27, every player with a contract. "
                "The line is the market price of a win. Players below the line produce more wins than they are paid "
                "for. Highlighted: the nine largest surpluses and deficits for next season."))
s.append(table([
    ["Most underpaid", "Age", "Years", "Salary $M", "WAR", "Surplus $M", "Most overpaid", "Age", "Years", "Salary $M",
     "WAR", "Surplus $M"],
    ["Gilgeous-Alexander", "28", "5", "314", "52.6", "+612", "Paolo Banchero", "24", "5", "241", "0.0", "−241"],
    ["Wembanyama", "23", "6", "269", "46.6", "+600", "Trae Young", "28", "4", "213", "2.0", "−180"],
    ["Amen Thompson", "24", "6", "220", "38.8", "+502", "Jaylen Brown", "30", "3", "183", "1.1", "−166"],
    ["Chet Holmgren", "25", "5", "241", "30.3", "+296", "Keyonte George", "23", "6", "162", "0.0", "−162"],
    ["Dyson Daniels", "24", "4", "100", "20.6", "+249", "Joel Embiid", "33", "3", "188", "3.2", "−138"],
    ["Derrick White", "32", "3", "98", "17.4", "+181", "Walker Kessler", "25", "4", "130", "0.8", "−116"],
    ["Kon Knueppel", "21", "3", "36", "13.2", "+179", "Ayo Dosunmu", "27", "5", "112", "0.0", "−112"],
    ["Christian Braun", "26", "5", "125", "17.0", "+174", "Damian Lillard", "36", "4", "118", "0.9", "−104"],
], [2.75 * cm, 0.8 * cm, 0.95 * cm, 1.35 * cm, 1.0 * cm, 1.5 * cm, 2.75 * cm, 0.8 * cm, 0.95 * cm, 1.35 * cm,
    1.0 * cm, 1.5 * cm],
    "Table 3. Surplus over the whole remaining contract (salary, WAR and surplus summed over its years). Options are "
    "counted as guaranteed."))
s.append(P("The market price of a win for 2026-27 is about $14.7M. At that price only 31% of players under contract "
           "project to earn their salary next season, and 10 of the 27 players paid $45M or more. The largest "
           "surpluses in the league sit on young players on rookie-scale or early extension deals, and on players "
           "whose value comes from defense and efficiency: Derrick White, Dyson Daniels and Neemias Queta all return "
           "several times their salary. The largest deficits sit on high-usage scorers whose on-court impact is modest "
           "by RAPM, and on stars past 32 on the last years of big deals."))

s.append(P("4.5 Are three-point shooters overpaid?", h2))
s.append(P("For every player-season from 2011-12 on, salary is expressed in wins at that season's price and regressed on "
           "the projected WAR made the summer before, 3-point attempts per 36 minutes and 3-point percentage from the "
           "previous season (both as z-scores within the season), a rookie-contract flag and age. A positive "
           "coefficient on shooting means shooters are paid more than the wins they are expected to bring."))
s.append(figure(FIG / "threes.png", "Figure 6. Extra salary per season for one standard deviation more 3-point volume "
                "(blue) or accuracy (orange), at the same expected wins. Bands are 95% bootstrap intervals over "
                "players.", width=TEXT_W * 0.86))
s.append(table([
    ["$M per season, +1 SD, same expected wins", "2011-12 to 2014-15", "2015-16 to 2019-20", "2020-21 to 2025-26"],
    ["3PA per 36", "−0.40 [−0.7, −0.0]", "−0.45 [−0.8, −0.1]", "+1.51 [0.8, 2.1]"],
    ["3P%", "−0.33 [−0.7, −0.0]", "−0.02 [−0.3, 0.3]", "−0.44 [−0.9, 0.1]"],
    ["3PA per 36, with points per 36 as control", "−0.86 [−1.2, −0.6]", "−0.80 [−1.1, −0.4]", "−0.03 [−0.6, 0.5]"],
    ["Points per 36 (same model)", "+1.97 [1.6, 2.4]", "+2.01 [1.5, 2.5]", "+5.32 [4.8, 5.9]"],
], [6.2 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm], "Table 4. Pay for shooting beyond expected wins, by era, with 95% "
   "intervals."))
s.append(P("Until 2019 high-volume shooters were slightly underpaid for the wins they brought. Around 2019-20 the market "
           "caught up and then passed them: since 2020-21 one standard deviation more volume comes with about $1.5M a "
           "season extra at the same expected wins. Accuracy was never paid for. Once points per 36 minutes is in the "
           "model the shooting premium disappears, and points carry a large premium of their own ($5.3M per standard "
           "deviation). So the market pays for scoring volume, and threes are one way to score."))

s.append(P("4.6 Does defense win championships?", h2))
s.append(figure(FIG / "defense.png", "Figure 7. Left: points of game margin per point of regular-season rating edge, "
                "offense and defense, with 95% intervals (playoffs bootstrapped over series). Right: every playoff "
                "team since 2010-11; diagonals connect teams with the same net rating."))
s.append(P("Each team's regular-season offense and defense are measured against the league average, and the margin of "
           "every game is regressed on the two teams' offensive and defensive edges. In the regular season a point of "
           "offensive edge is worth 0.89 points of margin and a point of defensive edge 0.83. In the playoffs the "
           "numbers are 1.14 and 0.92; the difference, −0.22 (interval −0.50 to +0.05), points slightly the other way. "
           "The same holds for wins. Champions are no more defensive than other good teams: 10 of 16 were top five in "
           "offense, 9 of 16 in defense, and 9 of 16 were top three in net rating. Net rating wins championships, and "
           "it does not matter much which end it comes from. If anything a defensive rating predicts a little less, "
           "probably because it holds more luck, such as the opponents' shooting."))

# 5
s.append(P("5. Using it in a front office", h1))
s.append(P("5.1 Contract ceilings", h2))
s.append(P("Before a negotiation starts, the ceiling for a deal is the sum over its years of projected WAR times the "
           "price of a win, with the aging curve and the player's own injury projection built in. Anything below the "
           "ceiling is surplus for the team. Derrick White, for example, projects at 6.4 WAR next season, about $94M of "
           "wins on a $30M salary; at 32 the ceiling for each extra year falls quickly, which the curve prices "
           "directly."))
s.append(P("5.2 Trades and extensions", h2))
s.append(P("Surplus is the asset a team actually trades. Rookie-scale stars are the most valuable contracts in the league "
           "by a wide margin, and that surplus shrinks the moment an extension starts. Two players with the same "
           "reputation can differ by hundreds of millions in surplus, depending on age and years left."))
s.append(P("5.3 Pricing injury risk", h2))
s.append(P("Injury history predicts missed games and does not predict a worse player. The protection should therefore "
           "target availability: games-played incentives, partial guarantees and shorter terms for the high-risk "
           "group, whose players are expected to miss three times as many games as the low-risk group. The model gives "
           "a per-player expected games number for every negotiation."))
s.append(P("5.4 Return-to-play planning", h2))
s.append(P("After an ACL tear, plan for a weaker first 20 games, about half a point per 100 below the player's level, and "
           "set minutes and matchups accordingly. The first month back says little about the player or the deal. For "
           "other injury types there is no measurable rust once a player is cleared."))
s.append(P("5.5 Where the market is mispriced", h2))
s += bullets(["Scoring volume is priced well above what it adds in wins. Paying for points per 36 is the most common "
              "overpay in the data.",
              "Three-point volume is priced in since about 2019-20. Accuracy is still not paid for.",
              "Defense-first and efficient role players deliver the largest surpluses outside rookie deals, and MVP "
              "voters ignore defense entirely, which suggests the wider market does too.",
              "For the playoffs, build net rating the cheapest way. There is no premium for getting there with "
              "defense."])
s.append(P("5.6 Running it", h2))
s.append(P("The pipeline reruns every summer from free data, one script per step (appendix B). Each step has its own "
           "check: cross-validation for RAPM, a backtest for the projection and the injury model, and the team-wins "
           "check for WAR. A team can plug in better inputs without touching the rest: tracking data or a box-score "
           "prior for RAPM, its own medical data for the injury model, and its cap sheet for the contract "
           "mechanics."))

# 6
s.append(P("6. Limitations", h1))
s += bullets([
    "RAPM is noisy and cannot fully separate players who almost always share the floor. A box-score prior would "
    "stabilize it and is not included yet.",
    "The win scale depends on the calibration and the replacement level. A replacement team wins about 34 games here, "
    "a narrower spread than most public WAR models.",
    "Contracts: player and team options are treated as guaranteed, and cap mechanics (aprons, cap holds, trade "
    "matching) are not modeled. The price of a win is the market average, while a contender may rationally pay more "
    "at the margin.",
    "Injury data: two sources with different completeness, normalized within season. No data for 2020-21. Injuries in "
    "the playoffs show up at the next season's first report. Some injury groups are small (16 to 28 ACL cases per "
    "measure).",
    "The salary analyses show associations: timing of signings and contract years are not modeled, and the "
    "rookie-contract flag is approximate.",
    "Model results will disagree with reputation for some players, most visibly high-usage scorers. That is partly "
    "the point, and partly the noise in the first bullet."])

# appendix
s.append(PageBreak())
s.append(P("Appendix A. Technical details", h1))
s.append(table([
    ["Step", "Choice", "Validation"],
    ["Lineups", "starters inferred per period, sub names matched to ids with aliases and suffix rules; score fixes for "
     "rows that reset to 0-0", "97.9-99.9% of games kept per season"],
    ["RAPM", "ridge, 2 rows per stint (each side on offense), possession weights, home-court term; penalty by 5-fold "
     "GroupKFold over games", "CV error vs league-average baseline per season"],
    ["Value", "50/30/20 × minutes, 1,000 ghost minutes at 0, each season aged to next season's age", "backtest RMSE "
     "1.903 to 1.888 with aging"],
    ["Aging", "fixed effects (player + age dummies, ridge α=1), delta method as check, weighted quadratic",
     "peak 27; methods agree to about 33"],
    ["WAR", "calibration 1.70 × value − 0.54 on next-season team differentials; replacement −0.61 from bottom-20% "
     "salaries; 33.6 points per win", "team wins r = 0.78, MAE 6.2"],
    ["Injury risk", "normalized injury share, 3-year blend, age, minutes; linear, shrunk to the mean", "rolling "
     "backtest MAE 12.36 vs 12.88 (league average) vs 12.46 (boosting)"],
    ["Injury windows", "10+ games out; pre = last 82 games (after the previous return), post = first 20 / 82 games "
     "back; post windows fitted as change from pre (offset); controls at pivots in healthy stretches; separate fits for "
     "injuries and controls", "1,295 injuries, 2,544 controls; empirical-Bayes shrinkage, bootstrap intervals"],
    ["Price of a win", "Σ(salary − 20th percentile) / Σ positive projected WAR, per season; 9.1% yearly growth",
     "$13.5M in 2025-26"],
    ["Threes", "salary in WAR units ~ projected WAR + 3PA/36 + 3P% (shrunk to 35% with 100 attempts) + rookie flag "
     "+ age + age²", "player bootstrap, by era and season"],
    ["Defense", "game margin ~ O edge + D edge, regular-season ratings (leave-one-game-out in the regular season)",
     "series bootstrap for playoffs; logit on wins as check"],
], [2.5 * cm, 9.2 * cm, 5.1 * cm]))
s.append(P("Appendix B. Reproducing it", h1))
s.append(P("Everything runs from the repository with Python 3 (pandas, numpy, scipy, scikit-learn, matplotlib, "
           "pdfplumber). stats.nba.com blocks most cloud servers, so the roster and box-score pulls run from a normal "
           "machine. In order:"))
for line in ["check_api, ingest, lineups, rapm, players, aging, value, mvp",
             "injuries_pst, injury_reports (download, then --parse), scrape_salaries",
             "availability, injury, injury_types, injury_windows",
             "war, surplus, threes, playoffs, defense",
             "report/make_figures.py, report/build_report.py"]:
    s.append(P(line, mono))
s.append(Spacer(1, 6))
s.append(P("Data: play-by-play from github.com/shufinskiy/nba_data; stats.nba.com through nba_api; injury log from "
           "github.com/gboogy/nba-injury-data-scraper (ProSportsTransactions); injury reports from the NBA's official "
           "PDFs; salaries, contracts and MVP votes from basketball-reference.com.", caption))


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
                        bottomMargin=2.0 * cm, title="What is an NBA player worth?", author="Viktor Sadowski")
doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
print(f"report -> {OUT}")
