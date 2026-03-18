from __future__ import annotations

import html
import json
import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


from schemas import EvidenceItem, SourceType, TrialProfile
from utils import (
    extract_simple_outcome_signal,
    extract_simple_program_context,
    looks_like_broad_solid_tumor_trial,
    normalize_whitespace,
    safe_lower,
)

load_dotenv()


class BaseHTTPClient:
    def __init__(self) -> None:
        self.timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.user_agent = os.getenv("USER_AGENT", "trial-agent/1.0")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_text(self, url: str, params: Optional[Dict[str, Any]] = None) -> str:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text


class ClinicalTrialsClient(BaseHTTPClient):
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def search_trials(self, disease: str, drug: Optional[str], max_trials: int = 5) -> List[TrialProfile]:
        query_term = disease if not drug else f"{disease} AND {drug}"
        page_size = max(max_trials * 10, 30)
        params = {"query.term": query_term, "pageSize": page_size, "format": "json"}
        payload = self.get_json(self.BASE_URL, params=params)
        studies = payload.get("studies", [])
        ranked: List[Tuple[float, TrialProfile]] = []
        for study in studies:
            profile = self._study_to_profile(study)
            score = self._relevance_score(profile, disease, drug)
            ranked.append((score, profile))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [p for score, p in ranked if score > -5][:max_trials]

    def _relevance_score(self, profile: TrialProfile, disease: str, drug: Optional[str]) -> float:
        text_title = safe_lower(profile.title)
        text_condition = safe_lower(profile.condition)
        text_intervention = safe_lower(profile.intervention)
        disease_l = safe_lower(disease)
        drug_l = safe_lower(drug)
        score = 0.0

        if disease_l:
            if disease_l in text_title:
                score += 4.0
            if disease_l in text_condition:
                score += 5.0
            if "non-small cell lung cancer" in disease_l or "nsclc" in disease_l:
                if "nsclc" in text_title or "non-small cell lung cancer" in text_title:
                    score += 4.0
                if any(x in text_condition for x in ["nsclc", "non small cell lung cancer", "non-small cell lung cancer"]):
                    score += 5.0

        if drug_l:
            aliases = {drug_l}
            if drug_l == "pembrolizumab":
                aliases.add("keytruda")
            if any(a in text_title for a in aliases):
                score += 3.0
            if any(a in text_intervention for a in aliases):
                score += 5.0

        if profile.status and safe_lower(profile.status) in {"recruiting", "active, not recruiting", "completed"}:
            score += 1.0

        if looks_like_broad_solid_tumor_trial(" ".join([profile.title or "", profile.condition or ""])):
            score -= 4.0

        cond_count = len([x for x in (profile.condition or "").split(";") if x.strip()])
        if cond_count >= 5:
            score -= 2.0
        if cond_count >= 10:
            score -= 2.0
        return score

    def get_trial_detail(self, trial_id: str) -> EvidenceItem:
        params = {"query.id": trial_id, "pageSize": 1, "format": "json"}
        payload = self.get_json(self.BASE_URL, params=params)
        studies = payload.get("studies", [])
        if not studies:
            raise ValueError(f"No study found for trial_id={trial_id}")
        study = studies[0]
        profile = self._study_to_profile(study)
        return EvidenceItem(
            source_type=SourceType.REGISTRY,
            source_name="ClinicalTrials.gov",
            raw_text=json.dumps(study, ensure_ascii=False),
            extracted_fields={
                "trial_id": profile.trial_id,
                "title": profile.title,
                "condition": profile.condition,
                "intervention": profile.intervention,
                "sponsor": profile.sponsor,
                "phase": profile.phase,
                "status": profile.status,
                "brief_summary": profile.brief_summary,
            },
            confidence=0.92,
            url=f"https://clinicaltrials.gov/study/{trial_id}",
        )

    def _study_to_profile(self, study: Dict[str, Any]) -> TrialProfile:
        protocol = study.get("protocolSection", {}) or {}
        ident = protocol.get("identificationModule", {}) or {}
        status = protocol.get("statusModule", {}) or {}
        desc = protocol.get("descriptionModule", {}) or {}
        design = protocol.get("designModule", {}) or {}
        conditions = protocol.get("conditionsModule", {}) or {}
        arms = protocol.get("armsInterventionsModule", {}) or {}
        sponsor = protocol.get("sponsorCollaboratorsModule", {}) or {}

        nct_id = ident.get("nctId")
        title = ident.get("briefTitle") or ident.get("officialTitle")
        phase_list = design.get("phases") or []
        phase = ", ".join(phase_list) if phase_list else None
        overall_status = status.get("overallStatus")
        condition_list = conditions.get("conditions") or []
        intervention_list = [item.get("name") for item in (arms.get("interventions") or []) if item.get("name")]
        lead_sponsor = (sponsor.get("leadSponsor") or {}).get("name")
        brief_summary = desc.get("briefSummary")

        return TrialProfile(
            trial_id=normalize_whitespace(nct_id),
            title=normalize_whitespace(title),
            condition=normalize_whitespace("; ".join(condition_list) if condition_list else None),
            intervention=normalize_whitespace("; ".join(intervention_list) if intervention_list else None),
            sponsor=normalize_whitespace(lead_sponsor),
            phase=normalize_whitespace(phase),
            status=normalize_whitespace(overall_status),
            brief_summary=normalize_whitespace(brief_summary),
        )


class PubMedClient(BaseHTTPClient):
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self) -> None:
        super().__init__()
        self.tool = os.getenv("NCBI_TOOL", "trial_agent_prototype")
        self.email = os.getenv("NCBI_EMAIL", "email@gmail.com")
        self.api_key = os.getenv("NCBI_API_KEY", "")
        self.min_interval = 0.34 if not self.api_key else 0.11
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call_ts = time.time()

    def _base_params(self) -> Dict[str, str]:
        params = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def retrieve_publications(self, trial_profile: TrialProfile, max_records: int = 3) -> List[EvidenceItem]:
        queries = self._build_pubmed_queries(trial_profile)
        for query in queries:
            pmids = self._esearch(query, retmax=max_records * 4)
            if not pmids:
                continue
            articles = self._efetch(pmids)
            ranked = self._rank_articles(trial_profile, articles)
            ranked = [a for a in ranked if a[0] > 0][:max_records]
            if not ranked:
                continue
            evidence: List[EvidenceItem] = []
            for score, article in ranked:
                abstract_text = article.get("abstract") or ""
                title = article.get("title") or ""
                joined = normalize_whitespace(f"{title}. {abstract_text}") or ""
                outcome_signal, conf = extract_simple_outcome_signal(joined)
                evidence.append(
                    EvidenceItem(
                        source_type=SourceType.PUBLICATION,
                        source_name="PubMed",
                        raw_text=joined,
                        extracted_fields={
                            "outcome_signal": outcome_signal,
                            "publication_title": title,
                            "publication_summary": abstract_text[:1000] if abstract_text else None,
                            "pmid": article.get("pmid"),
                            "retrieval_query": query,
                            "match_score": score,
                        },
                        confidence=max(conf, min(0.9, 0.45 + 0.08 * score)),
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid')}/" if article.get("pmid") else None,
                    )
                )
            if evidence:
                return evidence
        return []

    def _build_pubmed_queries(self, trial_profile: TrialProfile) -> List[str]:
        q: List[str] = []
        disease = (trial_profile.condition or "").split(";")[0].strip()
        drug = (trial_profile.intervention or "").split(";")[0].strip()
        title = trial_profile.title or ""
        if trial_profile.trial_id:
            q.append(f'"{trial_profile.trial_id}"')
        if drug and disease:
            q.append(f'"{drug}" AND "{disease}"')
            q.append(f'"{drug}"[Title/Abstract] AND "{disease}"[Title/Abstract]')
        if title:
            q.append(f'"{title}"')
        if drug:
            q.append(f'"{drug}" AND clinical trial[Title/Abstract]')
        if disease:
            q.append(f'"{disease}" AND clinical trial[Title/Abstract]')
        deduped = []
        seen = set()
        for item in q:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _rank_articles(self, trial_profile: TrialProfile, articles: List[Dict[str, Optional[str]]]) -> List[Tuple[float, Dict[str, Optional[str]]]]:
        disease = safe_lower((trial_profile.condition or "").split(";")[0].strip())
        drug = safe_lower((trial_profile.intervention or "").split(";")[0].strip())
        trial_id = safe_lower(trial_profile.trial_id)
        ranked = []
        for article in articles:
            text = safe_lower(f"{article.get('title') or ''} {article.get('abstract') or ''}")
            score = 0.0
            if trial_id and trial_id in text:
                score += 8.0
            if drug and drug in text:
                score += 4.0
            if drug == "pembrolizumab" and "keytruda" in text:
                score += 2.0
            if disease and disease in text:
                score += 4.0
            if "non-small cell lung cancer" in disease or "nsclc" in disease:
                if "nsclc" in text or "non-small cell lung cancer" in text:
                    score += 3.0
            if any(k in text for k in ["phase", "randomized", "trial", "study"]):
                score += 1.0
            ranked.append((score, article))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def _esearch(self, term: str, retmax: int = 3) -> List[str]:
        self._throttle()
        params = {"db": "pubmed", "term": term, "retmax": str(retmax), "retmode": "json", **self._base_params()}
        time.sleep(0.4)
        data = self.get_json(self.ESEARCH_URL, params=params)
        return ((data.get("esearchresult") or {}).get("idlist") or [])

    def _efetch(self, pmids: List[str]) -> List[Dict[str, Optional[str]]]:
        self._throttle()
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", **self._base_params()}
        time.sleep(0.4)
        xml_text = self.get_text(self.EFETCH_URL, params=params)
        root = ET.fromstring(xml_text)
        articles: List[Dict[str, Optional[str]]] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = self._first_text(article, ".//PMID")
            title = self._collect_text(article.find(".//ArticleTitle"))
            abstract_parts = []
            for node in article.findall(".//Abstract/AbstractText"):
                label = node.attrib.get("Label")
                txt = self._collect_text(node)
                if txt:
                    abstract_parts.append(f"{label}: {txt}" if label else txt)
            abstract = normalize_whitespace(" ".join(abstract_parts))
            articles.append({
                "pmid": pmid,
                "title": normalize_whitespace(html.unescape(title or "")),
                "abstract": normalize_whitespace(html.unescape(abstract or "")),
            })
        return articles

    @staticmethod
    def _first_text(node: ET.Element, path: str) -> Optional[str]:
        child = node.find(path)
        if child is None or child.text is None:
            return None
        return normalize_whitespace(child.text)

    @staticmethod
    def _collect_text(node: Optional[ET.Element]) -> str:
        if node is None:
            return ""
        return "".join(node.itertext()).strip()


class SponsorPageClient(BaseHTTPClient):
    def __init__(self, sponsor_config_path: Optional[str] = None) -> None:
        super().__init__()
        self.config: Dict[str, Any] = {"by_drug": {}, "by_sponsor": {}}
        if sponsor_config_path:
            with open(sponsor_config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if "by_drug" in raw or "by_sponsor" in raw:
                self.config = raw
            else:
                # support flat mapping: {"pembrolizumab": [..]}
                self.config = {"by_drug": raw, "by_sponsor": {}}

    def retrieve_sponsor_evidence(self, trial_profile: TrialProfile, max_pages: int = 3) -> List[EvidenceItem]:
        urls = self._candidate_urls(trial_profile)[:max_pages]
        evidence_items: List[EvidenceItem] = []
        for url in urls:
            try:
                text = self._fetch_clean_text(url)
            except Exception:
                continue
            if not text:
                continue
            program_context, conf = extract_simple_program_context(text)
            scope, scope_reason, scope_score = self._classify_sponsor_scope(trial_profile, url, text)
            extracted = {
                "program_context": program_context,
                "sponsor_context_scope": scope,
                "sponsor_scope_reason": scope_reason,
                "sponsor_scope_score": scope_score,
            }
            evidence_items.append(
                EvidenceItem(
                    source_type=SourceType.SPONSOR,
                    source_name="Official Sponsor Page",
                    raw_text=text[:5000],
                    extracted_fields=extracted,
                    confidence=min(0.9, max(conf, scope_score)),
                    url=url,
                )
            )
        return evidence_items


    def _classify_sponsor_scope(self, trial_profile: TrialProfile, url: str, text: str) -> Tuple[str, str, float]:
        t = safe_lower(text)
        u = safe_lower(url)
        trial_id = safe_lower(trial_profile.trial_id)
        title = safe_lower(trial_profile.title)
        condition = safe_lower(trial_profile.condition)
        intervention = safe_lower(trial_profile.intervention)
        phase = safe_lower(trial_profile.phase)
        status = safe_lower(trial_profile.status)

        regimen_terms = [x.strip() for x in re.split(r"[;,]", intervention) if x.strip()]
        regimen_hits = sum(1 for term in regimen_terms[:5] if term and term in t)
        title_tokens = [w for w in re.findall(r"[a-z0-9-]+", title) if len(w) > 4 and w not in {"study", "treating", "patients", "phase", "trial", "randomized", "non-small", "small", "cancer", "lung"}]
        title_hits = sum(1 for tok in title_tokens[:8] if tok in t)
        condition_hit = int(any(x in t for x in ["nsclc", "non-small cell lung cancer", "non small cell lung cancer"])) if condition else 0
        drug_hit = int(any(term in t for term in regimen_terms[:3])) if regimen_terms else 0
        trial_id_hit = int(bool(trial_id and (trial_id in t or trial_id in u)))
        phase_hit = int(bool(phase and phase in t))
        status_hit = int(bool(status and status in t))

        # highly specific if the page mentions the NCT id or multiple regimen/title markers together
        if trial_id_hit:
            return "trial_program_specific", "Sponsor page explicitly references the trial identifier.", 0.9
        if regimen_hits >= 2 and (title_hits >= 1 or phase_hit or status_hit):
            return "trial_program_specific", "Sponsor page matches multiple regimen/title markers consistent with the current trial/program.", 0.82
        if title_hits >= 2 and (condition_hit or drug_hit):
            return "trial_program_specific", "Sponsor page closely matches the trial title and disease context.", 0.78

        if drug_hit and condition_hit:
            host = urlparse(url).netloc or "official sponsor site"
            return "drug_level", f"Sponsor page appears to be a broader {host} drug/indication page for the NSCLC setting rather than a trial-specific program page.", 0.65
        if drug_hit:
            return "drug_level", "Sponsor page supports drug-level continuity but does not clearly identify the current trial/program.", 0.58
        return "unclear", "Sponsor page relevance to the specific trial/program remains unclear.", 0.4

    def _candidate_urls(self, trial_profile: TrialProfile) -> List[str]:
        urls: List[str] = []
        drug_cfg = self.config.get("by_drug", {}) or {}
        sponsor_cfg = self.config.get("by_sponsor", {}) or {}
        intervention_text = safe_lower(trial_profile.intervention)
        for key, arr in drug_cfg.items():
            key_l = safe_lower(key)
            if key_l and key_l in intervention_text:
                urls.extend(arr)
            if key_l == "pembrolizumab" and "keytruda" in intervention_text:
                urls.extend(arr)
            if key_l == "keytruda" and "pembrolizumab" in intervention_text:
                urls.extend(arr)
        if trial_profile.sponsor:
            urls.extend(sponsor_cfg.get(trial_profile.sponsor, []))
            urls.extend(sponsor_cfg.get(safe_lower(trial_profile.sponsor), []))
        seen = set()
        out = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def _fetch_clean_text(self, url: str) -> str:
        html_text = self.get_text(url)
        soup = BeautifulSoup(html_text, "lxml")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        return normalize_whitespace(text) or ""
