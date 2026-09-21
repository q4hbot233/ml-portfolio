"""A shuttle passenger assistant that refuses - the interactive version.

Runs in the browser under stlite (Streamlit on Pyodide). Unlike most of the pages in
this showcase, this one does not only re-plot exported tables: it fits the classifier
in your browser at page load, on the training split shipped in ``data/``, and every
routing decision you see is that model running on your words. No pickle is shipped -
sklearn pickles break across versions - and no network call is made after the page
loads.

Layout rule for this module: all loading and computation lives in plain functions that
take arguments and return values, with no Streamlit call inside them. The Streamlit
section at the bottom reads the widgets, calls those functions and renders. That way
the compute layer can be exercised from ordinary Python.

What is shipped, and what each thing is allowed to prove:

* ``clinc_track_b_train.csv`` / ``clinc_track_b_test.csv`` - the full CLINC150 train and
  test splits, re-labelled onto the shuttle handler taxonomy (Track B). The utterances
  are other people's; the mapping is mine. Every accuracy number on this page is
  measured on these.
* ``domain_utterances.csv`` - 290 shuttle-specific phrasings I wrote myself (Track C).
  They are used as *training* data only, never as an evaluation set, because scoring a
  model on utterances written by the person who defined the intents measures that
  person's phrasing consistency and nothing else.
* ``gtfs/`` - Avon Transit's published timetable, which is what the handlers answer from.

The transit layer and the handlers are ports of ``src/gtfs.py`` and ``src/handlers.py``
from the repository, trimmed to what this page uses.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SEED = 20260918
REJECT = "reject"
SAFETY_LABEL = "safety_emergency"

#: The repository's chosen operating point, selected on the 150-way problem.
REPO_TAU = 0.2833333333333333
SAFETY_TAU = 0.15

INK = "#E9ECF1"
MUTED = "#99A0AC"
ACCEPT = "#6B9BD8"
REFUSE = "#E5484D"
GREY = "#7D8594"
WARM = "#F0A04B"
BG = "#08090B"
GRID = "#232830"

PLOT_STYLE = {
    "figure.dpi": 100, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "text.color": INK, "axes.titlecolor": INK,
    "axes.labelcolor": MUTED, "axes.edgecolor": GRID, "xtick.color": MUTED,
    "ytick.color": MUTED, "legend.labelcolor": MUTED,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "legend.frameon": False,
    "xtick.labelsize": 8.8, "ytick.labelsize": 8.8, "figure.titlesize": 11.5,
}

DAY_COLUMNS = ("monday", "tuesday", "wednesday", "thursday",
               "friday", "saturday", "sunday")

#: What each stop is next to. My labelling of the feed's own stop names - Avon
#: Transit names most stops after the thing they serve, so the amenity information
#: is already in the feed and all that is added here is the category word.
STOP_CATEGORIES: dict[str, tuple[str, ...]] = {
    "American National Bank": ("bank",),
    "Aspens": ("lodging",),
    "Avon Station": ("transit",),
    "Beaver Creek Village": ("resort", "dining", "shopping"),
    "Chapel Square/Mtn Family Health": ("health", "shopping", "dining"),
    "Christie Lodge": ("lodging",),
    "Christy Sports": ("shopping", "ski"),
    "City Market": ("grocery", "pharmacy"),
    "Comfort Inn": ("lodging",),
    "Elk Lot": ("parking",),
    "Loaded Joes": ("cafe", "dining"),
    "Northside Kitchen/Urgent Care": ("dining", "health"),
    "Sheraton Mountain Vista": ("lodging",),
    "Spring Hill Suites": ("lodging",),
    "Walmart": ("shopping", "grocery", "pharmacy"),
    "Westgate Plaza": ("shopping",),
}

CATEGORY_WORDS: dict[str, str] = {
    "eat": "dining", "food": "dining", "restaurant": "dining", "dinner": "dining",
    "lunch": "dining", "breakfast": "dining", "hungry": "dining", "bar": "dining",
    "drink": "dining", "coffee": "cafe", "cafe": "cafe", "grocer": "grocery",
    "groceries": "grocery", "supermarket": "grocery", "shop": "shopping",
    "shopping": "shopping", "store": "shopping", "mall": "shopping", "buy": "shopping",
    "pharmacy": "pharmacy", "chemist": "pharmacy", "prescription": "pharmacy",
    "doctor": "health", "medical": "health", "clinic": "health",
    "urgent care": "health", "hospital": "health", "hotel": "lodging",
    "stay": "lodging", "lodge": "lodging", "accommodation": "lodging",
    "bank": "bank", "atm": "bank", "cash": "bank", "park": "parking",
    "parking": "parking", "ski": "ski", "snowboard": "ski",
}

#: Terms that escalate on sight, before the classifier gets a vote. Word boundaries,
#: not substrings: with substring matching "fire" hits "fireplace" and "gun" hits
#: "begun". "help me" is deliberately absent - it escalated "can you help me plan a
#: trip to walmart", a false alarm on one of the most ordinary things a passenger says.
#:
#: The first group names the event. The second names what a passenger having one
#: types, which is a symptom: the original list held "heart attack" but not
#: "chest", so "my chest hurts" reached neither the word list nor the classifier's
#: safety class and was answered as an out-of-scope question. Symptom words are the
#: ones a person reaches for before they know what is happening to them, and they
#: are also the ones a transit vocabulary never uses, so they cost little here.
EMERGENCY_TERMS: tuple[str, ...] = (
    "emergency", "ambulance", "police", "911", "999", "fire", "smoke", "crash",
    "collision", "accident", "heart attack", "stroke", "seizure", "unconscious",
    "collapsed", "not breathing", "can't breathe", "cant breathe",
    "trouble breathing", "bleeding", "injured", "assault", "attacked", "weapon",
    "gun", "knife", "threatening", "harassing", "harassment", "unsafe", "overdose",
    "someone is hurt", "i'm hurt", "im hurt", "badly hurt",
    # symptoms, not events
    "chest pain", "chest hurts", "chest hurt", "chest tight", "chest tightness",
    "choking", "allergic", "anaphylactic", "epipen",
    "dizzy", "dizziness", "fainting", "feel faint", "passed out", "passing out",
    "stabbed", "seizing", "convulsing", "severe pain", "in labour", "in labor",
    # inflections the word-boundary match would otherwise miss: \bcrash\b does not
    # match "the vehicle has crashed", which is not a phrasing anyone would expect
    # this list to drop. Written out rather than stemmed, because a blanket suffix
    # rule turns "smoke" into "smoking" and escalates a question about the rules.
    "crashed", "crashing", "collided",
)
_EMERGENCY_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in EMERGENCY_TERMS) + r")\b")
_BARE_CRIES = {"help", "help me", "sos", "mayday", "help!!"}

_ROUTE_WORDS = {
    "red": "rt_red", "red line": "rt_red", "avon east": "rt_red",
    "blue": "rt_blue", "blue line": "rt_blue", "avon west": "rt_blue",
    "beaver creek express": "rt_beaver", "bc express": "rt_beaver",
    "express": "rt_beaver",
}
_FROM_TO = re.compile(r"\bfrom\s+(?P<origin>.+?)\s+(?:to|->|towards|toward)\s+"
                      r"(?P<destination>.+?)(?:\s*[?.!,]|$)", re.I)
_TO_FROM = re.compile(r"\bto\s+(?P<destination>.+?)\s+from\s+(?P<origin>.+?)"
                      r"(?:\s*[?.!,]|$)", re.I)
_BETWEEN = re.compile(r"\bbetween\s+(?P<origin>.+?)\s+and\s+(?P<destination>.+?)"
                      r"(?:\s*[?.!,]|$)", re.I)
#: "station" is NOT trimmed: trimming it turned "avon station" into "avon", which
#: prefix-matches both Avon Station and Avon Crossing and resolved to neither.
_STOPWORDS = {"the", "a", "an", "at", "in", "on", "my", "me", "here", "there",
              "please", "now", "today", "tonight", "go", "get"}

EARTH_RADIUS_MILES = 3958.7613
WALK_MPH = 3.0
MIN_TRANSFER_SECONDS = 120
DEFAULT_WALK_MILES = 0.35

#: Example questions, chosen to cover the three outcomes the page is about: routed and
#: answered, refused because no handler owns it, and refused because this taxonomy has
#: no handler for it at all.
EXAMPLES = [
    "how do i get from walmart to avon station",
    "where is the closest stop",
    "is there anywhere to eat near this stop",
    "when is the next shuttle",
    "is the shuttle delayed",
    "can you send a shuttle to pick me up",
    "i left my bag on the shuttle",
    "my chest hurts",
    "what is the exchange rate for euros",
]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def data_dir(base: str | Path | None = None) -> Path:
    """Where the shipped data lives, with a fallback for the browser bundle."""
    if base is not None:
        return Path(base)
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    for candidate in (here / "data", Path("data"), Path("/data")):
        if candidate.exists():
            return candidate
    return here / "data"


def load_utterances(path: str | Path) -> tuple[list[str], list[str]]:
    """A two-column ``text,label`` file as parallel lists."""
    texts: list[str] = []
    labels: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def load_repo_results(base: str | Path | None = None) -> dict:
    """The subset of ``reports/results.json`` this page quotes."""
    with open(data_dir(base) / "repo_results.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_corpora(base: str | Path | None = None) -> dict[str, tuple[list[str], list[str]]]:
    root = data_dir(base)
    return {
        "train": load_utterances(root / "clinc_track_b_train.csv"),
        "test": load_utterances(root / "clinc_track_b_test.csv"),
        "domain": load_utterances(root / "domain_utterances.csv"),
    }


# --------------------------------------------------------------------------- #
# the transit model - a port of src/gtfs.py, trimmed to what this page uses
# --------------------------------------------------------------------------- #
def parse_gtfs_time(value: str) -> int | None:
    """``'9:58:00'`` -> ``35880`` seconds after midnight of the service day.

    Hours of 24 or more are legal and preserved, so ``'25:10:00'`` is 90600, which is
    greater than any same-day time - exactly the ordering you want.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"not a GTFS time: {value!r}")
    hours, minutes, seconds = (int(p) for p in parts)
    if not (0 <= minutes < 60 and 0 <= seconds < 60):
        raise ValueError(f"not a GTFS time: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def format_time(seconds: int, twelve_hour: bool = True) -> str:
    seconds %= 24 * 3600
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    if not twelve_hour:
        return f"{hours:02d}:{minutes:02d}"
    suffix = "am" if hours < 12 else "pm"
    return f"{hours % 12 or 12}:{minutes:02d} {suffix}"


def format_duration(seconds: int) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def walk_minutes(miles: float) -> int:
    return max(1, round(miles / WALK_MPH * 60))


@dataclass(frozen=True)
class Stop:
    stop_id: str
    name: str
    lat: float
    lon: float
    wheelchair_boarding: str = ""

    @property
    def step_free(self) -> bool:
        """GTFS: 1 = accessible boarding, 2 = not, 0/'' = unknown."""
        return self.wheelchair_boarding == "1"


@dataclass(frozen=True)
class Route:
    route_id: str
    short_name: str
    long_name: str

    @property
    def name(self) -> str:
        # This feed ships ' Beaver Creek Express' with a leading space.
        return (self.long_name or self.short_name).strip()


@dataclass(frozen=True)
class FareProduct:
    name: str
    amount: float
    currency: str

    @property
    def is_free(self) -> bool:
        return self.amount == 0.0


@dataclass(frozen=True)
class Service:
    service_id: str
    days: tuple[int, ...]
    start_date: str
    end_date: str

    def runs_on(self, day: date) -> bool:
        stamp = day.strftime("%Y%m%d")
        return self.start_date <= stamp <= self.end_date and day.weekday() in self.days


@dataclass(frozen=True)
class Trip:
    trip_id: str
    route_id: str
    service_id: str
    wheelchair_accessible: str = ""


@dataclass(frozen=True)
class StopTime:
    trip_id: str
    stop_id: str
    sequence: int
    arrival: int | None
    departure: int | None
    headsign: str = ""


@dataclass(frozen=True)
class Departure:
    stop: Stop
    route: Route
    trip_id: str
    departure: int
    headsign: str

    def describe(self) -> str:
        head = f" towards {self.headsign}" if self.headsign else ""
        return f"{self.route.name}{head} at {format_time(self.departure)}"


@dataclass(frozen=True)
class Leg:
    route: Route
    board: Stop
    alight: Stop
    depart: int
    arrive: int
    trip_id: str
    n_stops: int


@dataclass(frozen=True)
class Itinerary:
    legs: list[Leg]

    @property
    def depart(self) -> int:
        return self.legs[0].depart

    @property
    def arrive(self) -> int:
        return self.legs[-1].arrive

    @property
    def duration(self) -> int:
        return self.arrive - self.depart

    @property
    def n_transfers(self) -> int:
        return len(self.legs) - 1


def _normalise(text: str) -> str:
    keep = [c.lower() if c.isalnum() else " " for c in text]
    return " ".join("".join(keep).split())


def _compact(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalnum())


@dataclass
class Feed:
    """An in-memory GTFS feed with the queries the handlers need."""

    agency_name: str
    timezone: str
    stops: dict[str, Stop]
    routes: dict[str, Route]
    services: dict[str, Service]
    trips: dict[str, Trip]
    stop_times: list[StopTime]
    agency_phone: str = ""
    agency_url: str = ""
    fares: tuple[FareProduct, ...] = ()

    _by_trip: dict[str, list[StopTime]] = field(default_factory=dict, repr=False)
    _by_stop: dict[str, list[StopTime]] = field(default_factory=dict, repr=False)
    _continues_into: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        by_trip: dict[str, list[StopTime]] = defaultdict(list)
        by_stop: dict[str, list[StopTime]] = defaultdict(list)
        for st in self.stop_times:
            by_trip[st.trip_id].append(st)
            by_stop[st.stop_id].append(st)
        for seq in by_trip.values():
            seq.sort(key=lambda s: s.sequence)
        self._by_trip = dict(by_trip)
        self._by_stop = dict(by_stop)
        self._continues_into = self._infer_blocks()

    def _infer_blocks(self) -> dict[str, str]:
        """Which trip each trip continues into, when the feed does not say.

        Both town routes are loops cut into 30-minute trips that chain end to end:
        trip N arrives at 7:00 and trip N+1 departs at 7:00 from the same stop on the
        same route. GTFS has ``block_id`` to say "this is one vehicle carrying on" and
        this feed omits it. Without that, a planner treats the loop anchor as a
        transfer, applies a buffer, misses the departure happening in that same minute
        and waits a full headway - a wrong answer, not a rounding error.
        """
        chains: dict[str, str] = {}
        starts: dict[tuple, list[str]] = defaultdict(list)
        for trip_id, seq in self._by_trip.items():
            if not seq or seq[0].departure is None:
                continue
            trip = self.trips[trip_id]
            starts[(trip.route_id, trip.service_id,
                    seq[0].stop_id, seq[0].departure)].append(trip_id)
        for trip_id, seq in self._by_trip.items():
            if not seq or seq[-1].arrival is None:
                continue
            trip = self.trips[trip_id]
            key = (trip.route_id, trip.service_id, seq[-1].stop_id, seq[-1].arrival)
            following = [t for t in starts.get(key, ()) if t != trip_id]
            if len(following) == 1:
                chains[trip_id] = following[0]
        return chains

    # -- lookup ------------------------------------------------------------ #

    def stop_names(self) -> list[str]:
        return sorted(s.name for s in self.stops.values())

    def find_stop(self, query: str) -> Stop | None:
        """Exact, then separator-insensitive, then prefix, then substring.

        No fuzzy matching on purpose: at 27 stops a bad fuzzy match is worse than
        admitting the name was not recognised, because the passenger then gets
        confidently sent to the wrong place.
        """
        if not query:
            return None
        needle = _normalise(query)
        if not needle:
            return None
        stops = list(self.stops.values())
        for stop in stops:
            if _normalise(stop.name) == needle:
                return stop
        # The feed writes the stop as "Walmart" but the headsign as "Wal-Mart", so a
        # passenger echoing the screen would otherwise match nothing.
        compact = _compact(query)
        for stop in stops:
            if _compact(stop.name) == compact:
                return stop
        prefix = [s for s in stops if _normalise(s.name).startswith(needle)]
        if len(prefix) == 1:
            return prefix[0]
        contains = [s for s in stops if needle in _normalise(s.name)]
        if len(contains) == 1:
            return contains[0]
        if len(needle) >= 4:
            reverse = [s for s in stops if _normalise(s.name) in needle]
            if len(reverse) == 1:
                return reverse[0]
        return None

    def route_for_trip(self, trip_id: str) -> Route:
        return self.routes[self.trips[trip_id].route_id]

    # -- calendar ---------------------------------------------------------- #

    def active_services(self, day: date) -> set[str]:
        return {sid for sid, svc in self.services.items() if svc.runs_on(day)}

    def active_trips(self, day: date) -> set[str]:
        live = self.active_services(day)
        return {tid for tid, t in self.trips.items() if t.service_id in live}

    def active_routes(self, day: date) -> list[Route]:
        ids = {self.trips[t].route_id for t in self.active_trips(day)}
        return sorted((self.routes[r] for r in ids), key=lambda r: r.name)

    def service_window(self) -> tuple[date, date]:
        """The calendar window the whole feed covers."""
        starts = [s.start_date for s in self.services.values()]
        ends = [s.end_date for s in self.services.values()]
        fmt = "%Y%m%d"
        return (datetime.strptime(min(starts), fmt).date(),
                datetime.strptime(max(ends), fmt).date())

    # -- queries ----------------------------------------------------------- #

    def _is_last_stop(self, st: StopTime) -> bool:
        return st.sequence == self._by_trip[st.trip_id][-1].sequence

    def departures_from(self, stop: Stop, day: date, after: int = 0, limit: int = 3,
                        route_id: str | None = None) -> list[Departure]:
        live = self.active_trips(day)
        out: list[Departure] = []
        for st in self._by_stop.get(stop.stop_id, ()):
            if st.trip_id not in live or st.departure is None or st.departure < after:
                continue
            if self._is_last_stop(st):
                continue                      # arrivals are not departures
            route = self.route_for_trip(st.trip_id)
            if route_id and route.route_id != route_id:
                continue
            out.append(Departure(stop, route, st.trip_id, st.departure, st.headsign))
        out.sort(key=lambda d: (d.departure, d.route.name))
        return out[:limit]

    def service_span(self, day: date, route_id: str | None = None
                     ) -> tuple[int, int] | None:
        live = self.active_trips(day)
        times: list[int] = []
        for st in self.stop_times:
            if st.trip_id not in live:
                continue
            if route_id and self.trips[st.trip_id].route_id != route_id:
                continue
            for t in (st.arrival, st.departure):
                if t is not None:
                    times.append(t)
        return (min(times), max(times)) if times else None

    def nearest_stops(self, lat: float, lon: float, limit: int = 3
                      ) -> list[tuple[Stop, float]]:
        ranked = sorted(((s, haversine_miles(lat, lon, s.lat, s.lon))
                         for s in self.stops.values()), key=lambda p: p[1])
        return ranked[:limit]

    def stops_near(self, stop: Stop, max_miles: float) -> list[tuple[Stop, float]]:
        out = [(s, haversine_miles(stop.lat, stop.lon, s.lat, s.lon))
               for s in self.stops.values() if s.stop_id != stop.stop_id]
        out = [(s, d) for s, d in out if d <= max_miles]
        out.sort(key=lambda p: p[1])
        return out

    # -- trip planning ----------------------------------------------------- #

    def plan_trip(self, origin: Stop, destination: Stop, day: date,
                  depart_after: int = 0, max_transfers: int = 2,
                  min_transfer_seconds: int = MIN_TRANSFER_SECONDS) -> Itinerary | None:
        """Earliest arrival from `origin` to `destination` on that service day.

        Round-based (RAPTOR-style): round 0 finds everything reachable on one vehicle,
        round 1 everything reachable with one transfer. The objective is arrival time,
        not hop count - a two-hop ride leaving in 40 minutes is worse for a passenger
        than a three-hop one leaving now. The round boundaries are what make "stayed on
        the same vehicle" and "changed vehicles" different things; relaxing in place
        across one pass reported a single through-ride as two legs with an impossible
        zero-minute transfer between them.
        Trips are iterated in sorted order rather than set order. Two itineraries can
        tie on arrival time - board later and ride fewer stops, or board now and ride
        more - and iterating a set let the winner depend on string hash order, so the
        same question could get two different (both correct) answers on two page loads.
        """
        if origin.stop_id == destination.stop_id:
            return None
        live = sorted(self.active_trips(day))
        arrival: dict[str, int] = {origin.stop_id: depart_after}
        parent: dict[str, Leg] = {}
        frontier: dict[str, tuple[int, str | None]] = {
            origin.stop_id: (depart_after, None)}

        for round_index in range(max_transfers + 1):
            improved: dict[str, tuple[int, str | None]] = {}
            for trip_id in live:
                sequence = self._by_trip.get(trip_id, ())
                route = self.route_for_trip(trip_id)
                board_index: int | None = None
                for i, st in enumerate(sequence):
                    if board_index is not None:
                        reach = st.arrival if st.arrival is not None else st.departure
                        if reach is not None and reach < arrival.get(st.stop_id, 1 << 30):
                            boarding = sequence[board_index]
                            arrival[st.stop_id] = reach
                            improved[st.stop_id] = (reach, trip_id)
                            parent[st.stop_id] = Leg(
                                route=route, board=self.stops[boarding.stop_id],
                                alight=self.stops[st.stop_id],
                                depart=boarding.departure, arrive=reach,
                                trip_id=trip_id, n_stops=i - board_index)
                    if st.departure is None or self._is_last_stop(st):
                        continue
                    entry = frontier.get(st.stop_id)
                    if entry is None:
                        continue
                    ready, arrived_on = entry
                    staying_on = (arrived_on is not None
                                  and self._continues_into.get(arrived_on) == trip_id)
                    buffer = 0 if (round_index == 0 or staying_on) else min_transfer_seconds
                    if ready + buffer > st.departure:
                        continue
                    if board_index is None:
                        board_index = i
            if not improved:
                break
            frontier = improved

        if destination.stop_id not in parent:
            return None
        legs: list[Leg] = []
        cursor = destination.stop_id
        for _ in range(max_transfers + 2):
            leg = parent.get(cursor)
            if leg is None:
                break
            legs.append(leg)
            cursor = leg.board.stop_id
        if not legs or cursor != origin.stop_id:
            return None
        legs.reverse()
        return Itinerary(_merge_through_legs(legs))


def _merge_through_legs(legs: list[Leg]) -> list[Leg]:
    """Collapse "got off and got straight back on the same vehicle" into one leg.

    Telling somebody to change at Buffalo Ridge West when they never leave their seat
    is a worse answer than the timetable deserves.
    """
    if len(legs) < 2:
        return legs
    merged = [legs[0]]
    for leg in legs[1:]:
        last = merged[-1]
        through = (last.route.route_id == leg.route.route_id
                   and last.alight.stop_id == leg.board.stop_id
                   and last.arrive == leg.depart)
        if through:
            merged[-1] = Leg(route=last.route, board=last.board, alight=leg.alight,
                             depart=last.depart, arrive=leg.arrive,
                             trip_id=last.trip_id, n_stops=last.n_stops + leg.n_stops)
        else:
            merged.append(leg)
    return merged


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_feed(base: str | Path | None = None) -> Feed:
    """Read the shipped GTFS text files into a :class:`Feed`."""
    root = Path(base) if base is not None else data_dir() / "gtfs"
    agency_rows = _rows(root / "agency.txt")
    agency = agency_rows[0] if agency_rows else {}

    stops = {r["stop_id"]: Stop(
        stop_id=r["stop_id"], name=r["stop_name"].strip(),
        lat=float(r["stop_lat"]), lon=float(r["stop_lon"]),
        wheelchair_boarding=(r.get("wheelchair_boarding") or "").strip())
        for r in _rows(root / "stops.txt")}

    routes = {r["route_id"]: Route(
        route_id=r["route_id"],
        short_name=(r.get("route_short_name") or "").strip(),
        long_name=(r.get("route_long_name") or "").strip())
        for r in _rows(root / "routes.txt")}

    services = {r["service_id"]: Service(
        service_id=r["service_id"],
        days=tuple(i for i, col in enumerate(DAY_COLUMNS) if r.get(col) == "1"),
        start_date=r["start_date"], end_date=r["end_date"])
        for r in _rows(root / "calendar.txt")}

    trips = {r["trip_id"]: Trip(
        trip_id=r["trip_id"], route_id=r["route_id"], service_id=r["service_id"],
        wheelchair_accessible=(r.get("wheelchair_accessible") or "").strip())
        for r in _rows(root / "trips.txt")}

    stop_times = [StopTime(
        trip_id=r["trip_id"], stop_id=r["stop_id"], sequence=int(r["stop_sequence"]),
        arrival=parse_gtfs_time(r.get("arrival_time", "")),
        departure=parse_gtfs_time(r.get("departure_time", "")),
        headsign=(r.get("stop_headsign") or "").strip())
        for r in _rows(root / "stop_times.txt")
        if r["trip_id"] in trips and r["stop_id"] in stops]

    fares = tuple(FareProduct(
        name=(r.get("fare_product_name") or "").strip(),
        amount=float(r.get("amount") or 0.0),
        currency=(r.get("currency") or "USD").strip())
        for r in _rows(root / "fare_products.txt"))

    return Feed(agency_name=agency.get("agency_name", "").strip(),
                timezone=agency.get("agency_timezone", "UTC").strip(),
                agency_phone=(agency.get("agency_phone") or "").strip(),
                agency_url=(agency.get("agency_url") or "").strip(),
                fares=fares, stops=stops, routes=routes, services=services,
                trips=trips, stop_times=stop_times)


def service_date_for(feed: Feed, today: date) -> tuple[date, bool]:
    """A date inside the feed's calendar window, and whether it is really today.

    The feed covers 31 March to 25 November 2026. A visitor arriving outside that window
    would otherwise be told nothing runs, which is true of the file and useless as a
    demonstration, so the day is pulled into the window on the same weekday and the page
    says which date it is answering for.
    """
    start, end = feed.service_window()
    if start <= today <= end:
        return today, True
    if today > end:
        # The latest date in the window that falls on today's weekday.
        candidate = end - timedelta(days=(end.weekday() - today.weekday()) % 7)
    else:
        candidate = start + timedelta(days=(today.weekday() - start.weekday()) % 7)
    if not (start <= candidate <= end):
        candidate = end if today > end else start
    return candidate, False


# --------------------------------------------------------------------------- #
# slot filling - a port of the string matching in src/handlers.py
# --------------------------------------------------------------------------- #
def find_route(text: str, feed: Feed) -> str | None:
    lowered = f" {text.lower()} "
    hits = [(len(p), rid) for p, rid in _ROUTE_WORDS.items() if f" {p} " in lowered]
    for route in feed.routes.values():
        for name in (route.name, route.short_name):
            if name and f" {name.strip().lower()} " in lowered:
                hits.append((len(name), route.route_id))
    return max(hits)[1] if hits else None


def _scan_for_stops(text: str, feed: Feed) -> list[Stop]:
    """Stop names occurring in the text, in order of appearance, longest match first."""
    lowered = text.lower()
    found: list[tuple[int, Stop]] = []
    taken: list[tuple[int, int]] = []
    for stop in sorted(feed.stops.values(), key=lambda s: -len(s.name)):
        needle = stop.name.lower()
        start = lowered.find(needle)
        if start < 0:
            compact = re.sub(r"[^a-z0-9]+", r"[^a-z0-9]*", needle)
            m = re.search(compact, lowered)
            if not m:
                continue
            start, end = m.span()
        else:
            end = start + len(needle)
        if any(start < b and a < end for a, b in taken):
            continue
        taken.append((start, end))
        found.append((start, stop))
    found.sort(key=lambda pair: pair[0])
    return [stop for _, stop in found]


def _trim(fragment: str) -> str:
    words = [w for w in re.split(r"\s+", fragment.strip().lower()) if w]
    while words and words[0] in _STOPWORDS:
        words.pop(0)
    while words and words[-1] in _STOPWORDS:
        words.pop()
    return " ".join(words)


def _resolve_fragment(fragment: str, feed: Feed) -> Stop | None:
    fragment = fragment.strip()
    if not fragment:
        return None
    for candidate in (fragment, _trim(fragment)):
        if candidate:
            stop = feed.find_stop(candidate)
            if stop is not None:
                return stop
    scanned = _scan_for_stops(fragment, feed)
    return scanned[0] if scanned else None


def find_stops(text: str, feed: Feed) -> tuple[Stop | None, Stop | None]:
    """(origin, destination) named in the utterance, or (None, None).

    Returns nothing rather than guessing: a stop the passenger did not ask for is
    worse than admitting the name was not recognised, because they will act on it.
    """
    for pattern in (_FROM_TO, _TO_FROM, _BETWEEN):
        match = pattern.search(text)
        if not match:
            continue
        origin = _resolve_fragment(match.group("origin"), feed)
        destination = _resolve_fragment(match.group("destination"), feed)
        if origin or destination:
            return origin, destination
    mentioned = _scan_for_stops(text, feed)
    if len(mentioned) >= 2:
        return mentioned[0], mentioned[1]
    if len(mentioned) == 1:
        only = mentioned[0]
        name = only.name.lower()
        before = text.lower().split(name, 1)[0] if name in text.lower() else ""
        if re.search(r"\b(to|towards|toward|reach|get to|going to)\s*$", before.strip()):
            return None, only
        return only, None
    return None, None


def find_category(text: str) -> str | None:
    lowered = text.lower()
    for word, category in sorted(CATEGORY_WORDS.items(), key=lambda kv: -len(kv[0])):
        if word in lowered:
            return category
    return None


@dataclass(frozen=True)
class Context:
    """Everything the handlers know beyond the words themselves."""

    day: date
    seconds: int
    lat: float | None = None
    lon: float | None = None

    @property
    def has_location(self) -> bool:
        return self.lat is not None and self.lon is not None


def _resolve_origin(text: str, feed: Feed, ctx: Context) -> tuple[Stop | None, str]:
    origin, destination = find_stops(text, feed)
    stop = origin or destination
    if stop is not None:
        return stop, "named"
    if ctx.has_location:
        nearest = feed.nearest_stops(ctx.lat, ctx.lon, limit=1)
        if nearest:
            return nearest[0][0], "nearest"
    return None, "unknown"


def _contact(feed: Feed) -> str:
    bits = [b for b in (feed.agency_phone, feed.agency_url) if b]
    return " / ".join(bits) if bits else feed.agency_name


def _join(items) -> str:
    items = list(items)
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------- #
# handlers - every sentence is assembled from the timetable, not generated
# --------------------------------------------------------------------------- #
def _no_stop() -> str:
    return ("I need to know which stop you mean. Tell me the stop name, or share your "
            "location and I will use the closest one.")


def h_next_departure(text: str, feed: Feed, ctx: Context) -> str:
    stop, how = _resolve_origin(text, feed, ctx)
    if stop is None:
        return _no_stop()
    route_id = find_route(text, feed)
    departures = feed.departures_from(stop, ctx.day, after=ctx.seconds, limit=3,
                                      route_id=route_id)
    prefix = f"From {stop.name}" if how == "named" else f"Your nearest stop is {stop.name}"
    if not departures:
        running = feed.active_routes(ctx.day)
        if route_id and route_id not in {r.route_id for r in running}:
            return (f"{feed.routes[route_id].name} is not running today. Today's routes "
                    f"are {_join([r.name for r in running])}.")
        span = feed.service_span(ctx.day)
        tail = f" Service today ran until {format_time(span[1])}." if span else ""
        return f"{prefix}, there are no more departures today.{tail}"
    lines = [f"- {d.describe()} ({format_duration(d.departure - ctx.seconds)} away)"
             for d in departures]
    return f"{prefix}, the next departures are:\n" + "\n".join(lines)


def h_trip_planning(text: str, feed: Feed, ctx: Context) -> str:
    origin, destination = find_stops(text, feed)
    if destination is None and origin is not None:
        origin, destination = None, origin
    if destination is None:
        return ("Tell me where you are going and I will plan it. I know "
                f"{len(feed.stops)} stops on this network.")
    if origin is None:
        origin, _ = _resolve_origin("", feed, ctx)
    if origin is None:
        return _no_stop()
    if origin.stop_id == destination.stop_id:
        return f"You are already at {destination.name}."
    itinerary = feed.plan_trip(origin, destination, ctx.day, depart_after=ctx.seconds)
    if itinerary is None:
        return (f"I could not find a way from {origin.name} to {destination.name} on "
                "today's timetable. It may be on a route that is not running today, or "
                "there may be nothing left this evening.")
    lines = [f"- {l.route.name}: board at {l.board.name} {format_time(l.depart)}, ride "
             f"{l.n_stops} stop{'s' if l.n_stops != 1 else ''} to {l.alight.name}, "
             f"arriving {format_time(l.arrive)}" for l in itinerary.legs]
    transfers = ("direct" if itinerary.n_transfers == 0
                 else f"{itinerary.n_transfers} change"
                      f"{'s' if itinerary.n_transfers > 1 else ''}")
    return (f"{origin.name} to {destination.name}, {transfers}, "
            f"{format_duration(itinerary.duration)}:\n" + "\n".join(lines))


def h_nearest_stop(text: str, feed: Feed, ctx: Context) -> str:
    if not ctx.has_location:
        return ("I need your location to find the closest stop. You can also name a "
                "landmark and I will tell you the stop that serves it.")
    ranked = feed.nearest_stops(ctx.lat, ctx.lon, limit=3)
    closest, miles = ranked[0]
    others = ", ".join(f"{s.name} ({walk_minutes(m)} min walk)" for s, m in ranked[1:])
    served = feed.departures_from(closest, ctx.day, after=ctx.seconds, limit=1)
    tail = (f" Next departure {format_time(served[0].departure)} on "
            f"{served[0].route.name}." if served
            else " Nothing more departs from there today.")
    return (f"Your closest stop is {closest.name}, about {walk_minutes(miles)} minutes' "
            f"walk ({miles:.2f} mi).{tail}"
            + (f" Also nearby: {others}." if others else ""))


def h_poi_near_stop(text: str, feed: Feed, ctx: Context) -> str:
    stop, how = _resolve_origin(text, feed, ctx)
    if stop is None:
        return _no_stop()
    category = find_category(text)
    candidates: list[tuple[Stop, float, tuple[str, ...]]] = []
    here = STOP_CATEGORIES.get(stop.name, ())
    if here:
        candidates.append((stop, 0.0, here))
    for other, miles in feed.stops_near(stop, DEFAULT_WALK_MILES):
        cats = STOP_CATEGORIES.get(other.name, ())
        if cats:
            candidates.append((other, miles, cats))
    if category:
        candidates = [c for c in candidates if category in c[2]]
    if not candidates:
        what = f" for {category}" if category else ""
        return (f"I do not have anything listed{what} within walking distance of "
                f"{stop.name}. What I know about is limited to the places the network "
                "names its stops after.")
    lines = [f"- {p.name} ({'/'.join(c)}) - "
             + ("right at your stop" if m == 0 else f"{walk_minutes(m)} min walk")
             for p, m, c in candidates[:4]]
    what = f"{category} " if category else ""
    return f"Near {stop.name}, {what}options are:\n" + "\n".join(lines)


def h_service_hours(text: str, feed: Feed, ctx: Context) -> str:
    route_id = find_route(text, feed)
    running = feed.active_routes(ctx.day)
    if route_id and route_id not in {r.route_id for r in running}:
        return (f"{feed.routes[route_id].name} is not running today. Today the network "
                f"is running {_join([r.name for r in running])}.")
    span = feed.service_span(ctx.day, route_id=route_id)
    if span is None:
        return "Nothing is scheduled today."
    what = feed.routes[route_id].name if route_id else "Service"
    return (f"{what} today runs from {format_time(span[0])} to {format_time(span[1])}. "
            f"Routes running today: {_join([r.name for r in running])}.")


def h_fares(text: str, feed: Feed, ctx: Context) -> str:
    """Answered from ``fare_products.txt``, not from something I typed."""
    if not feed.fares:
        return ("The feed does not publish fare information, so I cannot tell you the "
                "price. Contact the operator.")
    if all(f.is_free for f in feed.fares):
        f = feed.fares[0]
        return (f"It is free. {feed.agency_name} publishes a fare of {f.amount:.2f} "
                f"{f.currency} - there is nothing to pay and no ticket to buy. "
                "Just board.")
    return "Fares:\n" + "\n".join(
        f"- {f.name}: {f.amount:.2f} {f.currency}" for f in feed.fares)


def h_accessibility(text: str, feed: Feed, ctx: Context) -> str:
    """Conservative: unknown is never reported as yes."""
    stop, how = _resolve_origin(text, feed, ctx)
    accessible = [t for t in feed.trips.values() if t.wheelchair_accessible == "1"]
    fleet = (f"All {len(feed.trips)} scheduled trips are marked wheelchair accessible "
             "in the timetable" if len(accessible) == len(feed.trips) else
             f"{len(accessible)} of {len(feed.trips)} scheduled trips are marked "
             "wheelchair accessible")
    contact = _contact(feed)
    if stop is None:
        unknown = [s.name for s in feed.stops.values() if not s.step_free]
        note = ("Every stop on the network is marked as having accessible boarding."
                if not unknown else
                f"{len(unknown)} stops are not marked accessible.")
        return (f"{fleet}. {note} For a ramp or boarding assistance at a specific stop, "
                f"tell me which stop, or call ahead: {contact}.")
    if stop.step_free:
        head = f"{stop.name} is marked as having accessible boarding, and {fleet.lower()}."
    else:
        # The feed's 0 means "unknown", not "no". Saying yes on unknown strands somebody.
        head = (f"The timetable does not record accessible boarding at {stop.name}, "
                "which means unknown rather than no.")
    return (f"{head} If you need the ramp deployed or help boarding, call {contact} and "
            "they will arrange it.")


def h_service_disruption(text: str, feed: Feed, ctx: Context) -> str:
    """Says what it cannot see: this is the static timetable, not a live feed."""
    stop, how = _resolve_origin(text, feed, ctx)
    running = feed.active_routes(ctx.day)
    scheduled = ""
    if stop is not None:
        departures = feed.departures_from(stop, ctx.day, after=ctx.seconds, limit=2)
        scheduled = (" The timetable has " + _join([d.describe() for d in departures])
                     + "." if departures else
                     f" Nothing further is scheduled from {stop.name} today.")
    return ("I only have the published timetable - I cannot see where the vehicles "
            f"actually are, so I cannot confirm a delay.{scheduled} Routes scheduled "
            f"today: {_join([r.name for r in running])}. For live status, contact "
            f"{_contact(feed)}.")


def h_lost_item(text: str, feed: Feed, ctx: Context) -> str:
    return (f"I cannot search the vehicles myself. Report it to {feed.agency_name} - "
            f"{_contact(feed)} - with the route, the stop you boarded at and roughly "
            "when you travelled, and they can check the vehicle.")


def h_ride_request(text: str, feed: Feed, ctx: Context) -> str:
    """No on-demand product exists in this feed, so this says no, clearly."""
    stop, how = _resolve_origin(text, feed, ctx)
    suggestion = ""
    if stop is not None:
        departures = feed.departures_from(stop, ctx.day, after=ctx.seconds, limit=1)
        if departures:
            suggestion = (f" The nearest scheduled service to you is "
                          f"{departures[0].describe()} from {stop.name}.")
    return ("This is a fixed-route service - I cannot send a vehicle to you or book a "
            f"seat.{suggestion} For door-to-door transport, contact {_contact(feed)}.")


def h_safety_emergency(text: str, feed: Feed, ctx: Context) -> str:
    """The conservative path. It escalates and it does not try to be clever."""
    return ("If anyone is in danger, call 911 now. I am not a substitute for emergency "
            f"services. To reach {feed.agency_name} directly: {_contact(feed)}. If you "
            "are on board, use the emergency intercom or the marked emergency stop, and "
            "stay where staff can see you.")


def h_reject(text: str, feed: Feed, ctx: Context) -> str:
    """Rejection is a behaviour, not an error, so it is a handler like any other.

    The line about 911 is here rather than in the safety handler on purpose. The
    safety handler only runs when something recognised the question as an
    emergency; this one runs when nothing did, which is exactly the case that
    needs the line. A word list assembled by one person will always be missing a
    phrasing, and the cost of printing one extra sentence to a passenger asking
    about fares is not comparable to the cost of not printing it to the other one.
    """
    return ("That is outside what I can help with. **If this is a medical emergency or "
            "anyone is in danger, call 911 now.** Otherwise: I can answer questions about "
            "shuttle departures, planning a trip between stops, the nearest stop, what is "
            "near a stop, service hours, fares, and accessibility.")


HANDLER_FUNCTIONS = {
    "next_departure": h_next_departure, "trip_planning": h_trip_planning,
    "nearest_stop": h_nearest_stop, "poi_near_stop": h_poi_near_stop,
    "service_hours": h_service_hours, "fares": h_fares,
    "accessibility": h_accessibility, "service_disruption": h_service_disruption,
    "lost_item": h_lost_item, "ride_request": h_ride_request,
    "safety_emergency": h_safety_emergency, REJECT: h_reject,
}


def run_handler(intent: str, text: str, feed: Feed, ctx: Context) -> str:
    return HANDLER_FUNCTIONS.get(intent, h_reject)(text, feed, ctx)


# --------------------------------------------------------------------------- #
# the classifier - fitted in the browser, not unpickled
# --------------------------------------------------------------------------- #
def build_model(texts: list[str], labels: list[str]):
    """TF-IDF char_wb 3-5 grams into a linear head, fitted by SGD.

    This is the ``tfidf_char`` rung from the repository's ladder, with the repository's
    hyperparameters and seed. It is not the rung the repository selected for Track B -
    that was a linear SVM, which scored a shade higher - but an SVM has a margin and not
    a probability, and a threshold study needs something to threshold. The repository
    made the same choice for the same reason and ran its rejection sweep on this rung.

    char_wb keeps n-grams inside word boundaries, which is what makes it robust to typos
    without turning the utterance into mush.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  sublinear_tf=True, min_df=1, lowercase=True,
                                  strip_accents="unicode")),
        ("clf", SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=40, tol=1e-4,
                              random_state=SEED)),
    ]).fit(texts, labels)


def training_set(corpora: dict, include_domain: bool) -> tuple[list[str], list[str]]:
    """The rows the model is fitted on.

    ``include_domain`` adds the 290 utterances I wrote. That is what makes the assistant
    usable - CLINC150 contains no shuttle vocabulary at all - and it is also why nothing
    measured on those utterances appears anywhere on this page.
    """
    texts, labels = list(corpora["train"][0]), list(corpora["train"][1])
    if include_domain:
        texts += list(corpora["domain"][0])
        labels += list(corpora["domain"][1])
    return texts, labels


def mentions_emergency(text: str) -> bool:
    """Literal term match, no model involved.

    Crude on purpose. Its job is not to be the emergency detector - the classifier is -
    but to be the one path that cannot be broken by a training distribution or by a
    threshold chosen on a validation set of a hundred rows.
    """
    stripped = text.lower().strip().rstrip("!.?")
    if stripped in _BARE_CRIES:
        return True
    return _EMERGENCY_PATTERN.search(text.lower()) is not None


def score_texts(model, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(classes, per-class score matrix). Rows sum to one; this is a ranking, not a
    calibrated probability, and the threshold is swept empirically rather than read off
    as a confidence."""
    return np.asarray(model.classes_), np.asarray(model.predict_proba(texts))


def route(classes: np.ndarray, scores: np.ndarray, tau: float, text: str,
          use_safety_keywords: bool = True, safety_tau: float = SAFETY_TAU) -> dict:
    """The policy, for one utterance. The order is the design.

    1. Safety keywords, before the classifier gets a vote. A model that is 99% accurate
       is still wrong once in a hundred times, and this is the one route where being
       wrong is not recoverable by the passenger asking again.
    2. The model's safety class, at its own much lower threshold, so phrasings the word
       list does not contain still escalate.
    3. The explicit reject class.
    4. The confidence threshold.
    5. The handler.
    """
    order = np.argsort(scores)[::-1]
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else top
    best_label, best_score = str(classes[top]), float(scores[top])
    runner_up, runner_up_score = str(classes[second]), float(scores[second])
    base = {"confidence": best_score, "argmax": best_label,
            "runner_up": runner_up, "runner_up_confidence": runner_up_score,
            "accepted": True}

    if use_safety_keywords and mentions_emergency(text):
        return {**base, "intent": SAFETY_LABEL, "confidence": 1.0,
                "reason": "keyword_safety",
                "explain": "escalated on an emergency term before the model ran"}

    where = np.where(classes == SAFETY_LABEL)[0]
    if len(where):
        safety_score = float(scores[int(where[0])])
        if safety_score >= safety_tau:
            return {**base, "intent": SAFETY_LABEL, "confidence": safety_score,
                    "reason": "model_safety",
                    "explain": f"escalated: safety score {safety_score:.2f} is over the "
                               f"safety threshold of {safety_tau:.2f}"}

    if best_label == REJECT:
        return {**base, "intent": REJECT, "reason": "reject_class", "accepted": False,
                "explain": f"refused: the model's own reject class won at "
                           f"{best_score:.3f}; best handler was {runner_up} at "
                           f"{runner_up_score:.3f}"}
    if best_score < tau:
        return {**base, "intent": REJECT, "reason": "threshold", "accepted": False,
                "explain": f"refused: best guess was {best_label} at {best_score:.3f}, "
                           f"under the threshold of {tau:.2f}"}
    return {**base, "intent": best_label, "reason": "model",
            "explain": f"routed to {best_label} at {best_score:.3f} "
                       f"(runner-up {runner_up} at {runner_up_score:.3f})"}


# --------------------------------------------------------------------------- #
# evaluation - all of it on real utterances, none of it on mine
# --------------------------------------------------------------------------- #
def evaluation_arrays(classes: np.ndarray, scores: np.ndarray,
                      truth: list[str]) -> dict:
    """Everything the tau sweep needs, computed once."""
    pred = classes[scores.argmax(axis=1)]
    conf = scores.max(axis=1)
    y = np.asarray(truth, dtype=object)
    return {"pred": pred, "conf": conf, "truth": y,
            "in_scope": y != REJECT, "out_of_scope": y == REJECT}


def metrics_at_tau(arrays: dict, tau: float) -> dict:
    """In-scope accuracy and out-of-scope recall, the ``oos_class`` mechanism.

    Rejection means "the argmax is the reject class OR the confidence is below tau", so
    tau = 0 is the pure explicit-class mechanism and raising tau layers thresholding on
    top. That is the same definition the repository's curve uses, which is what makes
    the two comparable.

    "In scope" here means the utterance belongs to a shuttle handler; "out of scope"
    means no handler owns it. Both sets are real CLINC150 utterances.
    """
    pred, conf = arrays["pred"], arrays["conf"]
    y, ins, oos = arrays["truth"], arrays["in_scope"], arrays["out_of_scope"]
    refused = (pred == REJECT) | (conf < tau)
    correct = (pred == y) & ~refused
    n_in, n_oos = int(ins.sum()), int(oos.sum())
    return {
        "tau": float(tau),
        "in_scope_accuracy": float(correct[ins].mean()) if n_in else float("nan"),
        "in_scope_acceptance": float((~refused[ins]).mean()) if n_in else float("nan"),
        "oos_recall": float(refused[oos].mean()) if n_oos else float("nan"),
        "false_alarm_rate": float((~refused[oos]).mean()) if n_oos else float("nan"),
        "joint_accuracy": float((correct[ins].sum() + refused[oos].sum()) / len(y)),
        "n_in_scope": n_in, "n_out_of_scope": n_oos,
    }


def live_curve(arrays: dict, n: int = 61) -> dict[str, np.ndarray]:
    taus = np.linspace(0.0, 1.0, n)
    rows = [metrics_at_tau(arrays, t) for t in taus]
    return {k: np.array([r[k] for r in rows])
            for k in ("tau", "in_scope_accuracy", "oos_recall", "joint_accuracy")}


def per_handler_recall(arrays: dict, tau: float, labels: list[str]) -> list[dict]:
    """Recall per handler at this tau, on real utterances only."""
    pred, conf = arrays["pred"], arrays["conf"]
    y = arrays["truth"]
    refused = (pred == REJECT) | (conf < tau)
    out = []
    for label in labels:
        mask = y == label
        support = int(mask.sum())
        if not support:
            continue
        if label == REJECT:
            hit = int(refused[mask].sum())
        else:
            hit = int(((pred == y) & ~refused)[mask].sum())
        out.append({"label": label, "recall": hit / support, "support": support,
                    "n_correct": hit})
    return out


def live_confusions(arrays: dict, tau: float, texts: list[str], k: int = 6) -> list[dict]:
    """The most frequent (true -> what actually happened) mistakes, with examples.

    A confusion matrix is not a result. This is: the handful of pairs the model collides,
    with an utterance from each, so a reader can judge whether the model is weak or the
    distinction is genuinely unavailable in the words.
    """
    pred, conf, y = arrays["pred"], arrays["conf"], arrays["truth"]
    refused = (pred == REJECT) | (conf < tau)
    outcome = np.where(refused, REJECT, pred)
    support = Counter(y.tolist())
    tally: dict[tuple[str, str], list[int]] = {}
    for i, (t, o) in enumerate(zip(y.tolist(), outcome.tolist())):
        if t != o:
            tally.setdefault((t, o), []).append(i)
    pairs = [{"true_label": t, "outcome": o, "count": len(rows),
              "rate": len(rows) / support[t],
              "examples": [texts[i] for i in rows[:2]]}
             for (t, o), rows in tally.items()]
    pairs.sort(key=lambda c: (-c["count"], c["true_label"], c["outcome"]))
    return pairs[:k]


def macro_f1_at_tau(arrays: dict, tau: float, labels: list[str]) -> float:
    """Macro-F1 over the handler taxonomy, with refusal folded in as the reject class."""
    pred, conf, y = arrays["pred"], arrays["conf"], arrays["truth"]
    outcome = np.where((pred == REJECT) | (conf < tau), REJECT, pred)
    scores = []
    for label in labels:
        tp = int(((outcome == label) & (y == label)).sum())
        fp = int(((outcome == label) & (y != label)).sum())
        fn = int(((outcome != label) & (y == label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def training_composition(labels: list[str]) -> list[tuple[str, int]]:
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda kv: (kv[0] == REJECT, -kv[1], kv[0]))


def n_reject_examples(labels: list[str]) -> int:
    return sum(1 for l in labels if l == REJECT)


def first_tau_that_bites(curve: dict[str, np.ndarray]) -> float:
    """The lowest tau at which in-scope accuracy has dropped at all.

    On this taxonomy the explicit reject class does nearly all the refusing, so the
    threshold is inert over a long stretch of its range and this says where it stops
    being inert.
    """
    acc = curve["in_scope_accuracy"]
    dropped = np.flatnonzero(acc < acc[0] - 1e-9)
    return float(curve["tau"][dropped[0]]) if len(dropped) else float(curve["tau"][-1])


# --------------------------------------------------------------------------- #
# figures - matplotlib, sized for a ~700px column
# --------------------------------------------------------------------------- #
def fig_rejection(repo_curve: dict, operating_point: dict, live: dict[str, np.ndarray],
                  tau: float, label: str):
    """Two panels: the repository's measured trade-off, and this page's own.

    They are not the same problem and the axes say so. The left is the 150-intent
    problem the repository selected its operating point on. The right is the
    deployment-shaped taxonomy this page runs, where the explicit reject class
    has already done nearly all the work before the threshold gets a turn.
    """
    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))

        pts = repo_curve["points"]
        x = [p["oos_recall"] for p in pts]
        y = [p["in_scope_accuracy"] for p in pts]
        ax1.plot(x, y, color=ACCEPT, lw=1.9, zorder=3)
        op = operating_point["test"]
        ax1.scatter([op["oos_recall"]], [op["in_scope_accuracy"]], s=46, color=REFUSE,
                    zorder=4)
        ax1.annotate(f"chosen $\\tau$ = {operating_point['tau']:.3f}\n"
                     f"{op['oos_recall']:.3f} recall, {op['in_scope_accuracy']:.3f} acc",
                     xy=(op["oos_recall"], op["in_scope_accuracy"]),
                     xytext=(0.06, 0.30), textcoords="axes fraction",
                     fontsize=8.2, color=REFUSE,
                     arrowprops=dict(arrowstyle="-", color=REFUSE, lw=0.8))
        ax1.set_xlabel("out-of-scope recall (share of unanswerable\nquestions correctly refused)")
        ax1.set_ylabel("in-scope accuracy")
        ax1.set_title("The repository's curve: 150 intents", fontsize=9.8)
        ax1.set_xlim(-0.03, 1.03)
        ax1.set_ylim(-0.03, 1.0)

        ax2.plot(live["oos_recall"], live["in_scope_accuracy"], color=ACCEPT, lw=1.9,
                 zorder=3)
        here = metrics_at_tau_from_curve(live, tau)
        ax2.scatter([here["oos_recall"]], [here["in_scope_accuracy"]], s=46,
                    color=REFUSE, zorder=4)
        ax2.annotate(f"your $\\tau$ = {tau:.2f}",
                     xy=(here["oos_recall"], here["in_scope_accuracy"]),
                     xytext=(0.05, 0.14), textcoords="axes fraction",
                     fontsize=8.2, color=REFUSE,
                     arrowprops=dict(arrowstyle="-", color=REFUSE, lw=0.8))
        ax2.set_xlabel("share of unanswerable questions refused")
        ax2.set_ylabel("in-scope accuracy")
        ax2.set_title(f"This page's curve: {label}", fontsize=9.8)
        lo = float(np.nanmin(live["oos_recall"]))
        pad = max(0.002, (1.0 - lo) * 0.08)
        ax2.set_xlim(lo - pad, 1.0 + pad)
        ax2.set_ylim(-0.03, 1.0)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))

        fig.suptitle("Refusing more costs accuracy - but only once the reject class "
                     "has run out of road", fontsize=10.6, fontweight="semibold", x=0.01,
                     ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def metrics_at_tau_from_curve(live: dict[str, np.ndarray], tau: float) -> dict:
    """Nearest point on a precomputed curve. Used only for drawing the marker."""
    i = int(np.argmin(np.abs(live["tau"] - tau)))
    return {k: float(v[i]) for k, v in live.items()}


def fig_per_handler(repo_rows: list[dict], live_rows: list[dict], tau: float):
    """Repository recall against this page's, handler by handler.

    Reject is on the chart rather than dropped: correctly refusing is the behaviour
    this whole page is about, and it is also the easiest row on it.
    """
    order = ([r["label"] for r in repo_rows if r["label"] != REJECT]
             + [r["label"] for r in repo_rows if r["label"] == REJECT])
    live_map = {r["label"]: r for r in live_rows}
    repo_map = {r["label"]: r for r in repo_rows}
    names = [("reject - no handler owns it" if l == REJECT else l.replace("_", " "))
             for l in order]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 0.52 * len(order) + 2.1))
        ypos = np.arange(len(order))
        ax.barh(ypos + 0.19, [repo_map[l]["recall"] for l in order], height=0.36,
                color=GREY, label="repository, linear SVM, no threshold")
        ax.barh(ypos - 0.19,
                [live_map[l]["recall"] if l in live_map else np.nan for l in order],
                height=0.36, color=ACCEPT,
                label=f"this page, fitted live, $\\tau$ = {tau:.2f}")
        for i, l in enumerate(order):
            ax.text(1.02, i, f"n={repo_map[l]['support']:,}", va="center",
                    fontsize=8.1, color=MUTED)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("recall on held-out CLINC150 utterances")
        ax.set_title("Trip planning and points of interest are where it loses people",
                     fontsize=10.2, pad=24)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, fontsize=8.3)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
    return fig


def fig_composition(rows: list[tuple[str, int]]):
    """What the model was fitted on, on a log axis because the imbalance is the point."""
    labels = [r[0].replace("_", " ") for r in rows]
    counts = [r[1] for r in rows]
    colors = [REFUSE if r[0] == REJECT else ACCEPT for r in rows]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(rows) + 1.7))
        ax.barh(np.arange(len(rows)), counts, color=colors, height=0.68)
        for i, c in enumerate(counts):
            ax.text(c * 1.12, i, f"{c:,}", va="center", fontsize=8.2, color=MUTED)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(8, max(counts) * 3.2)
        ax.set_xlabel("training utterances (log scale)")
        ax.set_title("Most of the training set is examples of what to refuse",
                     fontsize=10.2)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# streamlit layer - widgets, calls, rendering; no computation of its own
# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover - exercised by the browser, not by the tests
    import streamlit as st

    st.set_page_config(page_title="A shuttle assistant that refuses", page_icon="🚐",
                       layout="centered")

    @st.cache_data(show_spinner=False)
    def _corpora():
        return load_corpora()

    @st.cache_data(show_spinner=False)
    def _repo():
        return load_repo_results()

    @st.cache_resource(show_spinner=False)
    def _feed():
        return load_feed()

    @st.cache_resource(show_spinner="Fitting the classifier in your browser…")
    def _model(include_domain: bool):
        texts, labels = training_set(_corpora(), include_domain)
        return build_model(texts, labels)

    @st.cache_data(show_spinner=False)
    def _arrays(include_domain: bool):
        model = _model(include_domain)
        texts, truth = _corpora()["test"]
        classes, scores = score_texts(model, texts)
        return evaluation_arrays(classes, scores, truth)

    @st.cache_data(show_spinner=False)
    def _live_curve(include_domain: bool):
        return live_curve(_arrays(include_domain))

    repo = _repo()
    feed = _feed()
    corpora = _corpora()

    st.title("A shuttle assistant that refuses")
    ks = repo["track_c"]["keyword_safety"]
    caught = round(ks["recall_on_author_written_emergencies"] * ks["n_emergencies"])
    with st.container(border=True):
        st.caption("In short")
        st.markdown(
            "A public-data rebuild of the rider assistant from a Northwestern capstone I led "
            "for an autonomous shuttle operator. A TF-IDF classifier sends passenger "
            f"questions to {len(repo['handlers'])} handlers that answer from a real transit "
            "timetable, and refuses anything no handler owns. The emergency word list "
            f"catches {caught} of the {ks['n_emergencies']} emergency phrasings in the test "
            "set, so every refusal also tells the passenger to call 911.")
    st.markdown(
        "A passenger types a question. A classifier decides which handler owns it, and "
        "if nothing does, the assistant says so instead of guessing. The routing half is "
        "easy and everyone builds it. The refusing half is the one that decides whether "
        "the thing can go on a vehicle, because a router that is 95% accurate on the "
        "questions it was built for and confidently wrong on everything else answers "
        "*\"my chest hurts\"* with a bus timetable.\n\n"
        "**This page is not a recording.** The classifier is fitted in your browser, on "
        "the training split shipped with the page, when you load it. Everything below "
        "is that model running - on your words, and on 5,500 held-out utterances it has "
        "never seen.")

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Controls")
        build_choice = st.radio(
            "Training set",
            ["CLINC150 + my shuttle utterances", "CLINC150 only"],
            index=0,
            help="CLINC150 contains no shuttle vocabulary, so a model trained only on it "
                 "can route five handlers and refuses the rest. The second option shows "
                 "you exactly that.")
        include_domain = build_choice.startswith("CLINC150 +")

        tau = st.slider("Confidence threshold τ", 0.0, 0.99, float(REPO_TAU), 0.01,
                        help="Below this, the assistant refuses. "
                             "The default is the operating point the repository selected "
                             "on the 150-intent problem.")
        use_keywords = st.checkbox(
            "Emergency keyword escalation", value=True,
            help="A literal term match that fires before the classifier gets a vote.")

        st.divider()
        window_start, window_end = feed.service_window()
        default_day, is_really_today = service_date_for(feed, date.today())
        day = st.date_input("Service date", value=default_day,
                            min_value=window_start, max_value=window_end)
        clock = st.slider("Time of day", 0, 23, 9, 1, format="%d:00")
        stop_names = ["not shared"] + feed.stop_names()
        standing = st.selectbox(
            "Passenger is standing at", stop_names, index=0,
            help="This page has no GPS. Pick a stop to stand at and the location-aware "
                 "handlers will use that stop's published coordinates.")
        if not is_really_today:
            st.caption(f"Today is outside the feed's calendar "
                       f"({window_start:%-d %b %Y} to {window_end:%-d %b %Y}), so the "
                       f"date defaults to {default_day:%A %-d %B %Y}.")

    here = None if standing == "not shared" else next(
        s for s in feed.stops.values() if s.name == standing)
    ctx = Context(day=day, seconds=clock * 3600,
                  lat=here.lat if here else None, lon=here.lon if here else None)

    model = _model(include_domain)
    arrays = _arrays(include_domain)
    curve = _live_curve(include_domain)
    now = metrics_at_tau(arrays, tau)
    classes = np.asarray(model.classes_)
    taxonomy_labels = [r["label"] for r in repo["track_b"]["test"]["per_class"]]
    # The like-for-like comparison against results.json: same data, same taxonomy, no
    # threshold, and none of my own utterances in the training set.
    reference_macro_f1 = macro_f1_at_tau(_arrays(False), 0.0, taxonomy_labels)

    # -------------------------------------------------------------- 1. ask it
    st.subheader("1. Ask it something")
    if "question" not in st.session_state:
        st.session_state["question"] = EXAMPLES[0]
    st.caption("Or start from one of these:")
    for row_start in (0, 3, 6):
        cols = st.columns(3)
        for col, example in zip(cols, EXAMPLES[row_start:row_start + 3]):
            if col.button(example, key=f"ex{example}", use_container_width=True):
                st.session_state["question"] = example
    question = st.text_input("Your question", key="question")

    if question.strip():
        _, scores = score_texts(model, [question])
        decision = route(classes, scores[0], tau, question,
                         use_safety_keywords=use_keywords)
        left, right = st.columns([3, 2])
        left.metric("Handler", decision["intent"].replace("_", " "))
        right.metric("Confidence", f"{decision['confidence']:.3f}",
                     delta=f"{decision['confidence'] - tau:+.3f} vs τ",
                     delta_color="normal" if decision["accepted"] else "inverse")
        if decision["intent"] == SAFETY_LABEL:
            st.error(f"**Escalated.** {decision['explain']}.")
            st.markdown(run_handler(SAFETY_LABEL, question, feed, ctx))
        elif decision["accepted"]:
            st.success(f"**Answered.** {decision['explain']}.")
            st.markdown(run_handler(decision["intent"], question, feed, ctx))
        else:
            st.warning(f"**Refused.** {decision['explain']}.")
            st.markdown(run_handler(REJECT, question, feed, ctx))
        with st.expander("What every class scored"):
            order = np.argsort(scores[0])[::-1]
            st.dataframe(
                {"class": [str(classes[i]).replace("_", " ") for i in order],
                 "score": [round(float(scores[0][i]), 4) for i in order]},
                hide_index=True, use_container_width=True)
            st.caption(
                "Rows sum to one, but this is a ranking, not a calibrated probability. "
                "The threshold below is swept and chosen empirically for that reason.")
        st.caption(
            "Every sentence in the answer above is assembled from the timetable, not "
            "generated. A wrong departure time stated fluently is worse than a plain one "
            "stated correctly, and a phrasing layer can always be added on top of a "
            "layer that is right.")

    # ------------------------------------------------------------ 2. the τ
    st.subheader("2. The threshold is the whole design decision")
    st.markdown(
        "Refusing is not free. Every utterance the assistant refuses in order to catch "
        "the ones it cannot answer is also an utterance it *could* have answered. Both "
        "numbers below are recomputed live, at your τ, on 5,500 held-out CLINC150 "
        "utterances - "
        f"{now['n_in_scope']} that a shuttle handler genuinely owns and "
        f"{now['n_out_of_scope']:,} that none of them does.")
    c1, c2, c3 = st.columns(3)
    c1.metric("In-scope accuracy", f"{now['in_scope_accuracy']:.3f}",
              help=f"Of the {now['n_in_scope']} held-out utterances a handler owns, the "
                   "share routed to the right handler and not refused.")
    c2.metric("Out-of-scope recall", f"{now['oos_recall']:.3f}",
              help=f"Of the {now['n_out_of_scope']:,} held-out utterances no handler "
                   "owns, the share correctly refused.")
    c3.metric("False alarms", f"{now['false_alarm_rate']:.3f}",
              help="The other side of the same coin: unanswerable questions that got an "
                   "answer anyway.")
    st.pyplot(fig_rejection(repo["track_a_rejection_curve"], repo["operating_point"],
                            curve, tau,
                            "11 handlers" if include_domain else "5 handlers"))
    flat_to = first_tau_that_bites(curve)
    st.markdown(
        f"On the 150-intent problem the repository had to buy out-of-scope recall with "
        f"in-scope accuracy, and it chose τ = {repo['operating_point']['tau']:.3f} by a "
        f"rule fixed before the analysis: reach 0.60 out-of-scope recall, then take the "
        f"highest in-scope accuracy among the points that clear it. It got "
        f"{repo['operating_point']['test']['oos_recall']:.3f} on the test split "
        f"(95% CI "
        f"[{repo['operating_point']['test']['oos_recall_ci95'][0]:.3f}, "
        f"{repo['operating_point']['test']['oos_recall_ci95'][1]:.3f}]) - it missed, and "
        f"missing is what selecting a threshold on 100 rows buys.\n\n"
        f"At deployment shape the curve looks different, and that is the more useful "
        f"finding. Reject is an explicit class here, trained on "
        f"{n_reject_examples(training_set(corpora, include_domain)[1]):,} real "
        f"examples of questions this assistant should not answer, and it has already "
        f"done the work before any threshold applies: moving τ changes nothing at all "
        f"until about {flat_to:.2f}. Past that you are no longer catching out-of-scope "
        f"questions - you are refusing passengers.")

    # ------------------------------------------------- 3. where it goes wrong
    st.subheader("3. Where it goes wrong")
    repo_rows = [{"label": r["label"], "recall": r["recall"], "support": r["support"]}
                 for r in repo["track_b"]["test"]["per_class"]]
    live_rows = per_handler_recall(arrays, tau, taxonomy_labels)
    st.pyplot(fig_per_handler(repo_rows, live_rows, tau))
    st.caption(
        f"Grey is `reports/results.json`: the linear SVM the repository selected for "
        f"this taxonomy, scored once on the test split with no threshold "
        f"(macro-F1 {repo['track_b']['test']['macro_f1']:.4f}). Blue is the model this "
        f"page just fitted, at your τ. They are close, and where the blue bar is shorter "
        f"at a high τ that is the threshold refusing passengers, not the model getting "
        f"worse.")

    st.markdown("**The mistakes the model makes at your τ:**")
    confusions = live_confusions(arrays, tau, corpora["test"][0])
    if confusions:
        st.dataframe(
            {"asked about": [c["true_label"].replace("_", " ") for c in confusions],
             "what happened": [c["outcome"].replace("_", " ") for c in confusions],
             "count": [c["count"] for c in confusions],
             "share of that class": [f"{c['rate']:.1%}" for c in confusions],
             "an example": [c["examples"][0] for c in confusions]},
            hide_index=True, use_container_width=True)
    st.caption(
        "A 150x150 confusion matrix is not a result; this is. The repository's own top "
        "pair on this taxonomy is trip planning refused outright - "
        f"{repo['track_b']['top_confusions'][0]['count']} of 60, "
        f"{repo['track_b']['top_confusions'][0]['rate']:.1%} - on utterances like "
        f"*\"{repo['track_b']['top_confusions'][0]['examples'][0]}\"*, which is a "
        "distance question about a place this shuttle does not go.")

    with st.expander("What the model was fitted on"):
        texts, fit_labels = training_set(corpora, include_domain)
        st.pyplot(fig_composition(training_composition(fit_labels)))
        st.markdown(
            f"{len(texts):,} utterances, fitted in your browser when this page loaded. "
            "The six CLINC150 intents that map onto a shuttle handler are the only "
            "positives that are not mine:")
        st.dataframe(
            {"CLINC150 intent": list(repo["track_b"]["mapping"].keys()),
             "handler": [v.replace("_", " ")
                         for v in repo["track_b"]["mapping"].values()]},
            hide_index=True, use_container_width=True)
        st.markdown(
            "The interesting half of that table is the candidates thrown out. Each one "
            "looked like a match from the label name and stopped looking like one once "
            "the utterances were read:")
        st.dataframe(
            {"CLINC150 intent": [r["clinc_intent"]
                                 for r in repo["track_b"]["rejected_mappings"]],
             "an utterance": [r["example"] for r in repo["track_b"]["rejected_mappings"]],
             "why it is not mapped": [r["why_not"]
                                      for r in repo["track_b"]["rejected_mappings"]]},
            hide_index=True, use_container_width=True)

    # --------------------------------------------- 4. what is not evidence
    st.subheader("4. What this page is not evidence of")
    st.markdown(
        "**The shuttle utterances are mine, so nothing is measured on them.** With the "
        "default training set, 290 of the utterances the model learned from are "
        "shuttle-specific phrasings I wrote myself after defining the intents. That is "
        "the only reason the assistant can answer *\"when is the next shuttle\"* at all "
        "- CLINC150 is a general virtual-assistant corpus and contains no shuttle "
        "vocabulary. It also makes any score computed on those utterances circular: it "
        "would measure how consistently one person phrases things, not whether the model "
        "generalises to a passenger. `results.json` carries `circular: true` on that "
        "track as a field rather than a footnote, so the number cannot be pasted "
        "somewhere and lose the warning on the way. Its 5-fold macro-F1 of "
        f"{repo['track_c']['cv_macro_f1']['tfidf_char']:.4f} is not on this page as "
        "a result, and every accuracy above is measured on held-out CLINC150 utterances "
        "that I did not write.\n\n"
        "**Switch the training set to *CLINC150 only* in the sidebar** and you get the "
        "non-circular assistant in full: five handlers, and *\"when is the next "
        "shuttle\"*, *\"is the shuttle delayed\"* and *\"I left my bag on the shuttle\"* "
        "all refused, because no data anyone else collected has ever seen those "
        "questions. That is where a shuttle intent classifier built entirely from "
        "public data stands, and it is why the repository's conclusion is that "
        "the next step is a few hundred real passenger utterances, not a bigger model.\n\n"
        "**CLINC150 is not transit either.** It is crowdsourced English in which workers "
        "imagined a scenario and typed a sentence. Real passengers are terser, angrier, "
        "use local place names and type on a moving vehicle. Only six of its 150 intents "
        "map onto a shuttle handler at all. So the repository's headline "
        f"{repo['track_a']['test']['accuracy']:.4f} on the 150-way problem is evidence "
        "that the approach works and an upper bound on field performance.\n\n"
        "**And the emergency path is a word list, not a model.** It held *heart attack* "
        "but not *chest*, so *\"my chest hurts\"* matched nothing, the classifier did not "
        "reach the safety class either, and the assistant answered it as an out-of-scope "
        "question. Symptom terms and a few inflections are in the list now, but the "
        "measurement is the point: on the author-written emergencies it reaches "
        f"{repo['track_c']['keyword_safety']['recall_on_author_written_emergencies']:.0%} "
        "with a "
        f"{repo['track_c']['keyword_safety']['false_alarm_rate_on_real_utterances']:.2%} "
        "false-alarm rate on 5,500 real utterances - and that recall is on emergencies I "
        "wrote, which is exactly the circularity this section is about.\n\n"
        "The ten it still misses are *\"i need medical help\"*, *\"i'm being followed\"*, "
        "*\"there's a fight happening\"* - generic calls for help and security incidents, "
        "not medical phrasings. They could be written into the list in an afternoon, and "
        "they have not been, because a list tuned until it matches its own thirty test "
        "sentences measures nothing. What changed instead is the refusal: it now names "
        "911 before it lists what the assistant can do, so the phrasing nobody anticipated "
        "still reaches a passenger with the one instruction that matters. A word list "
        "assembled by one person will always be missing a phrasing; the fallback is what "
        "makes that survivable.")

    st.divider()
    st.caption(
        "**How it runs.** No pickle is shipped - sklearn pickles break across versions - "
        "so the vectoriser and the linear model are fitted here, in Pyodide, from the "
        "CSVs in `data/`. Nothing leaves your browser and no request is made after the "
        "page loads. The model is the `tfidf_char` rung of the repository's ladder "
        "(char_wb 3-5 grams, TF-IDF, a linear head fitted by SGD, seed "
        f"{SEED}), with the repository's hyperparameters. On the validation split it "
        f"reproduces `results.json` exactly: accuracy "
        f"{repo['track_b']['validation']['tfidf_char']['accuracy']:.4f}, macro-F1 "
        f"{repo['track_b']['validation']['tfidf_char']['macro_f1']:.4f}. The repository "
        f"selected a linear SVM for this taxonomy instead "
        f"(test macro-F1 {repo['track_b']['test']['macro_f1']:.4f} against this rung's "
        f"{reference_macro_f1:.4f}, both scored on the same 5,500 held-out utterances), "
        "because an SVM has a margin and not a probability, and a threshold study needs "
        "something to threshold.\n\n"
        "**Data.** CLINC150, used under CC BY 3.0 and redistributed here re-labelled onto "
        "the handler taxonomy: Stefan Larson, Anish Mahendran, Joseph J. Peper, "
        "Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin "
        "Leach, Michael A. Laurenzano, Lingjia Tang and Jason Mars, *An Evaluation "
        "Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP "
        f"2019. The timetable is Avon Transit's published GTFS feed - {feed.agency_name}, "
        f"Town of Avon, Colorado - used under CC BY 4.0: {repo['transit']['n_stops']} "
        f"stops, {repo['transit']['n_routes']} routes, {repo['transit']['n_trips']} "
        "trips. Avon Transit is a human-driven town shuttle; it is a real timetable to "
        "answer from, not evidence about autonomous-vehicle operations.\n\n"
        "**Provenance.** The repository is a clean-room rebuild informed by a ten-person "
        "graduate capstone for an autonomous shuttle operator whose operational data was "
        "not mine to publish. No sponsor code and no sponsor data is in it or on this "
        "page.")


if __name__ == "__main__":
    main()
