# -*- coding: utf-8 -*-
"""Core ASR alignment engine - shared between Tkinter and web app."""
import re
import numpy as np


def clean_word(w):
    return re.sub(r'[^\w]', '', w).lower()


def is_match(ref_word, hyp_word, ref_clean, hyp_clean):
    return (ref_clean == hyp_clean and ref_clean != '') or (ref_word == hyp_word)


def hirschberg_nw_score(ref_words, hyp_words, ref_clean, hyp_clean):
    n, m = len(ref_words), len(hyp_words)
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if is_match(ref_words[i-1], hyp_words[j-1], ref_clean[i-1], hyp_clean[j-1]):
                curr[j] = prev[j-1]
            else:
                curr[j] = min(prev[j], curr[j-1], prev[j-1]) + 1
        prev, curr = curr, [0] * (m + 1)
    return prev


def hirschberg(ref_words, hyp_words, ref_clean, hyp_clean, sentence_map):
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return [(None, hyp_words[j], 'ins', 0) for j in range(m)]
    if m == 0:
        return [(ref_words[i], None, 'del', sentence_map[i]) for i in range(n)]
    if n == 1:
        best_j, best_cost = 0, 1 + m
        for j in range(m):
            if is_match(ref_words[0], hyp_words[j], ref_clean[0], hyp_clean[j]):
                cost = j + (m - 1 - j)
                if cost < best_cost:
                    best_cost, best_j = cost, j
        result = []
        for j2 in range(m):
            if j2 == best_j and is_match(ref_words[0], hyp_words[j2], ref_clean[0], hyp_clean[j2]):
                result.append((ref_words[0], hyp_words[j2], 'match', sentence_map[0]))
            else:
                result.append((None, hyp_words[j2], 'ins', sentence_map[0]))
        return result

    mid = n // 2
    score_left = hirschberg_nw_score(ref_words[:mid], hyp_words, ref_clean[:mid], hyp_clean)
    score_right = hirschberg_nw_score(ref_words[mid:][::-1], hyp_words[::-1], ref_clean[mid:][::-1], hyp_clean[::-1])
    total = [score_left[j] + score_right[m - j] for j in range(m + 1)]
    j_split = total.index(min(total))
    left = hirschberg(ref_words[:mid], hyp_words[:j_split], ref_clean[:mid], hyp_clean[:j_split], sentence_map[:mid])
    right = hirschberg(ref_words[mid:], hyp_words[j_split:], ref_clean[mid:], hyp_clean[j_split:], sentence_map[mid:])
    return left + right


def generate_local_reason(error_words, missing_count, diffs, true_text, asr_text):
    reasons = []
    if len(error_words) > 0:
        reasons.append("Wrong word")
    if missing_count > 0:
        reasons.append("Missing word")
    insertion_count = sum(1 for c, _ in diffs if c == "blue")
    if insertion_count > 0:
        reasons.append("Extra word")
    true_punct = set(re.findall(r'[.,!?;:]', true_text))
    asr_punct = set(re.findall(r'[.,!?;:]', asr_text))
    if true_punct != asr_punct:
        if len(true_punct) > len(asr_punct):
            reasons.append("Missing punctuation")
        elif len(asr_punct) > len(true_punct):
            reasons.append("Extra punctuation")
        else:
            reasons.append("Punctuation error")
    return "; ".join(reasons) if reasons else ""


def run_alignment(true_lines, asr_lines):
    """Run ASR alignment and return structured results."""
    ref_norm, ref_disp, ref_sid = [], [], []
    for i, line in enumerate(true_lines):
        for w in line.split():
            ref_disp.append(w)
            ref_norm.append(re.sub(r"[^\w\s]", "", w.lower()))
            ref_sid.append(i)

    asr_norm, asr_disp, asr_line_ids = [], [], []
    for line_id, line in enumerate(asr_lines):
        for w in line.split():
            asr_disp.append(w)
            asr_norm.append(re.sub(r"[^\w\s]", "", w.lower()))
            asr_line_ids.append(line_id)

    n, m = len(ref_norm), len(asr_norm)

    # Choose algorithm based on size
    if n * m > 100_000_000:
        aligned_pairs = hirschberg(ref_disp, asr_disp, ref_norm, asr_norm, ref_sid)
    else:
        dp = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_norm[i-1] == asr_norm[j-1] and ref_norm[i-1] != '':
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

        i, j = n, m
        ops = []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref_norm[i-1] == asr_norm[j-1] and ref_norm[i-1] != '' else 1):
                op_type = "match" if ref_norm[i-1] == asr_norm[j-1] and ref_norm[i-1] != '' else "sub"
                ops.append((op_type, i-1, j-1))
                i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                ops.append(("del", i-1, None))
                i -= 1
            else:
                ops.append(("ins", None, j-1))
                j -= 1
        ops.reverse()

        # Convert ops to aligned_pairs format
        aligned_pairs = []
        for op, ri, hi in ops:
            if op == "match":
                aligned_pairs.append((ref_disp[ri], asr_disp[hi], 'match', ref_sid[ri]))
            elif op == "sub":
                aligned_pairs.append((ref_disp[ri], asr_disp[hi], 'sub', ref_sid[ri]))
            elif op == "del":
                aligned_pairs.append((ref_disp[ri], None, 'del', ref_sid[ri]))
            elif op == "ins":
                aligned_pairs.append((None, asr_disp[hi], 'ins', 0))

    # Build per-sentence results
    separated_rows = [[] for _ in true_lines]
    diff_rows = [[] for _ in true_lines]
    error_words = [[] for _ in true_lines]
    missing_counts = [0 for _ in true_lines]
    asr_line_to_sentence = {}
    last_sid = 0

    for ref_word, hyp_word, op, sid_hint in aligned_pairs:
        if op in ("match", "sub", "del"):
            sid = sid_hint
            last_sid = sid
        else:
            sid = last_sid

        if op == "match":
            word = hyp_word
            true_word = ref_word
            separated_rows[sid].append(word)
            true_n = re.sub(r"[^\w\s]", "", true_word.lower())
            asr_n = re.sub(r"[^\w\s]", "", word.lower())
            if true_n == asr_n and true_word != word:
                diff_rows[sid].append(("red", word))
                error_words[sid].append(word)
            else:
                diff_rows[sid].append(("black", word))
        elif op == "sub":
            word = hyp_word
            separated_rows[sid].append(word)
            diff_rows[sid].append(("red", word))
            error_words[sid].append(word)
        elif op == "del":
            missing = ref_word
            diff_rows[sid].append(("red", f"[{missing}]"))
            missing_counts[sid] += 1
        elif op == "ins":
            word = hyp_word
            separated_rows[sid].append(word)
            diff_rows[sid].append(("blue", word))

    # Build display texts from ASR line mapping
    # Re-derive ASR line to sentence mapping from aligned_pairs
    asr_word_idx = 0
    asr_line_to_sentence = {}
    last_sid2 = 0
    for ref_word, hyp_word, op, sid_hint in aligned_pairs:
        if op in ("match", "sub", "del"):
            sid = sid_hint
            last_sid2 = sid
        else:
            sid = last_sid2
        if op in ("match", "sub", "ins") and hyp_word is not None:
            if asr_word_idx < len(asr_line_ids):
                line_id = asr_line_ids[asr_word_idx]
                if line_id not in asr_line_to_sentence:
                    asr_line_to_sentence[line_id] = sid
            asr_word_idx += 1

    display_texts = []
    for i, true in enumerate(true_lines):
        display_lines = []
        for line_id, raw_line in enumerate(asr_lines):
            if line_id in asr_line_to_sentence and asr_line_to_sentence[line_id] == i:
                display_lines.append(raw_line)
        display_texts.append("\n".join(display_lines))

    # Build final alignment data
    alignment_data = []
    total_subs, total_dels, total_ins, total_refs = 0, 0, 0, 0

    for i, true in enumerate(true_lines):
        word_count = len(true.split())
        wrong_count = len(error_words[i]) + missing_counts[i]
        srr = 1 if wrong_count > 0 else 0
        score = 3 if wrong_count == 0 else ""

        separated_text = " ".join(separated_rows[i])
        s_subs = len(error_words[i])
        s_dels = missing_counts[i]
        s_ins = sum(1 for c, _ in diff_rows[i] if c == "blue")
        wer = ((s_subs + s_dels + s_ins) / word_count * 100) if word_count > 0 else 0

        total_subs += s_subs
        total_dels += s_dels
        total_ins += s_ins
        total_refs += word_count

        local_reason = generate_local_reason(
            error_words[i], missing_counts[i], diff_rows[i], true, separated_text
        )

        alignment_data.append({
            "id": i + 1,
            "true": true,
            "count": word_count,
            "asr_displayed": display_texts[i],
            "asr_separated": separated_text,
            "asr_nobreak": separated_text,
            "srr": srr,
            "wer": round(wer, 1),
            "wrong_count": wrong_count,
            "score": score,
            "diffs": diff_rows[i],
            "wrong_list": ",".join(error_words[i]),
            "translation": "",
            "ai_score": "",
            "ai_reason": local_reason
        })

    overall_wer = ((total_subs + total_dels + total_ins) / total_refs * 100) if total_refs > 0 else 0
    overall_stats = {"subs": total_subs, "dels": total_dels, "ins": total_ins, "refs": total_refs}

    return {
        "alignment_data": alignment_data,
        "overall_wer": round(overall_wer, 1),
        "overall_stats": overall_stats
    }


# --- Translation alignment helpers ---

def distribute_sentences_proportionally(sentences, asr_word_counts, total_segments):
    translations = []
    total_asr_words = sum(asr_word_counts)
    total_trans_words = sum(len(s.split()) for s in sentences)
    trans_idx = 0
    for i in range(total_segments):
        asr_ratio = asr_word_counts[i] / total_asr_words if total_asr_words > 0 else 1 / total_segments
        target_words = int(total_trans_words * asr_ratio)
        segment_trans = []
        current_words = 0
        while trans_idx < len(sentences) and (i == total_segments - 1 or current_words < target_words):
            sentence = sentences[trans_idx]
            words = len(sentence.split())
            if current_words > 0 and current_words + words > target_words * 1.5 and i < total_segments - 1:
                break
            segment_trans.append(sentence)
            current_words += words
            trans_idx += 1
        translations.append(" ".join(segment_trans) if segment_trans else "")
    return translations


def split_by_word_count(text, asr_word_counts, total_segments):
    words = text.split()
    total_words = len(words)
    total_asr_words = sum(asr_word_counts)
    translations = []
    word_idx = 0
    for i in range(total_segments):
        if total_asr_words > 0:
            ratio = asr_word_counts[i] / total_asr_words
            segment_word_count = max(1, int(total_words * ratio))
        else:
            segment_word_count = total_words // total_segments
        if i == total_segments - 1:
            segment_words = words[word_idx:]
        else:
            segment_words = words[word_idx:word_idx + segment_word_count]
        translations.append(" ".join(segment_words))
        word_idx += segment_word_count
    return translations


def distribute_fewer_sentences(sentences, total_segments, asr_word_counts):
    translations = []
    total_sentences = len(sentences)
    if total_sentences == 0:
        return [""] * total_segments
    total_asr_words = sum(asr_word_counts)
    sentence_idx = 0
    for i in range(total_segments):
        cumulative_words = sum(asr_word_counts[:i+1])
        target_sentence = int((cumulative_words / total_asr_words) * total_sentences) if total_asr_words > 0 else 0
        target_sentence = min(target_sentence, total_sentences - 1)
        if target_sentence > sentence_idx:
            sentence_idx = target_sentence
        translations.append(sentences[sentence_idx] if sentence_idx < total_sentences else "")
    return translations


def align_translation_local(alignment_data, trans_blob):
    """Align translation text to ASR segments using local methods."""
    total_asr_segments = len(alignment_data)
    asr_word_counts = [len(d["asr_nobreak"].split()) for d in alignment_data]
    line_splits = [l.strip() for l in trans_blob.split('\n') if l.strip()]

    if len(line_splits) == total_asr_segments:
        return line_splits, "line-by-line"

    sentence_splits = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$', trans_blob)
    sentence_splits = [s.strip() for s in sentence_splits if s.strip()]

    if len(sentence_splits) == total_asr_segments:
        return sentence_splits, "sentence-based"
    elif len(sentence_splits) > total_asr_segments:
        return distribute_sentences_proportionally(sentence_splits, asr_word_counts, total_asr_segments), "proportional"
    else:
        if len(sentence_splits) == 0:
            return split_by_word_count(trans_blob, asr_word_counts, total_asr_segments), "word-count based"
        else:
            return distribute_fewer_sentences(sentence_splits, total_asr_segments, asr_word_counts), "distributed"
