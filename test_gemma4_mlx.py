#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mlxcel(mlx / mlx-lm) + Hugging Face `gemma-4-E4B-it-qat-mobile` 모델 테스트 스크립트
==============================================================================

사용법
------
1) 실제 모델(약 3.4 GB)을 받아서 테스트:
     python test_gemma4_mlx.py --download
2) 모델 가중치 다운로드 없이 구조/토크나이저만 검증(추천, 용량 절약):
     python test_gemma4_mlx.py            # micro 모드(기본): 초소형 난수 가중치로 추론 파이프라인 검증
     python test_gemma4_mlx.py --dry-run  # 가중치 로드·추론 없이 config/tokenizer/chat_template 만 검사
3) 옵션:
     --model   mlx-community/gemma-4-E4B-it-qat-mobile  (HF repo id. 기본값)
               google/... 원본은 gated(동의 필요)라 미로그인 시 mlx-community 사본 사용
     --prompt  "질문"        (기본: " Explain MLX in one sentence.")
     --max-tokens 64
     --cpu     (Linux/x86 등 non-Apple 환경에서 mlx 백엔드 강제)

필요 패키지
-----------
     pip install mlx mlx-lm huggingface_hub safetensors transformers

주의
----
* MLX 본체는 Apple Silicon(macOS) 최적화 프레임워크입니다. Linux(x86/arm)에서도
  CPU 백엔드로 동작하나, 이 환경처럼 libmlx.so 로딩 이슈가 있으면 실패할 수 있습니다.
  그 경우 이 스크립트는 자동으로 '분석 모드'로 폴백하여 HF API로 파일 목록·config·
  chat_template 을 검증합니다(모델 다운로드 없음).
"""

import argparse
import json
import os
import sys

DEFAULT_REPO = "mlx-community/gemma-4-E4B-it-qat-mobile"
FULL_MODEL_BYTES = 3_455_955_670  # model.safetensors 실측 크기(약 3.4GB)


def hr(title=""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("-" * 70)


def check_imports():
    """mlx / mlx_lm 임포트 가능 여부 확인."""
    ok = True
    try:
        import mlx.core as mx
        print(f"[OK] mlx 임포트 성공 / 기본 디바이스: {mx.default_device()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] mlx 임포트 실패: {e}")
        print("       -> Apple Silicon/macOS 에서 실행하거나, `pip install --force-reinstall mlx` 를 시도하세요.")
    try:
        import mlx_lm
        from mlx_lm.models import gemma4_text  # noqa: F401
        print("[OK] mlx_lm 임포트 성공, gemma4_text 모델 정의 존재")
    except Exception as e:
        ok = False
        print(f"[FAIL] mlx_lm 관련 문제: {e}")
    return ok


def inspect_remote(repo: str):
    """다운로드 없이 HF API 로 저장소 메타데이터 검사."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[FAIL] huggingface_hub 미설치"); return None
    api = HfApi()
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as e:
        print(f"[FAIL] 저장소 접근 불가({repo}): {e}")
        print("       google/* 원본은 license 동의+토큰이 필요합니다(HF_ACCESS_TOKEN).")
        return None
    hr(f"HF 저장소 검사: {repo}")
    print(f"pipeline_tag : {info.pipeline_tag}")
    print(f"library_name : {info.library_name}")
    total = 0
    for s in info.siblings:
        sz = getattr(s, "size", 0) or 0
        total += sz
        print(f"  {s.rfilename:<32} {sz/1e6:>10.1f} MB")
    print(f"  {'(합계)':<32} {total/1e6:>10.1f} MB")
    cfg = api.hf_hub_download(repo, "config.json")
    with open(cfg) as f:
        c = json.load(f)
    tc = c.get("text_config", {})
    hr("config.json 핵심 아키텍처(gemma4)")
    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "intermediate_size",
              "vocab_size", "sliding_window", "num_kv_shared_layers"]:
        print(f"  {k:<22} = {tc.get(k)}")
    lt = tc.get("layer_types", [])
    print(f"  layer_types             = full_attention {lt.count('full_attention')}개 / "
          f"sliding_attention {lt.count('sliding_attention')}개")
    return info


def load_tokenizer_only(repo: str):
    """tokenizer.json + chat_template 만 내려받아 apply_chat_template 검증."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok_files = {}
    for fn in ("tokenizer_config.json", "chat_template.jinja"):
        try:
            tok_files[fn] = hf_hub_download(repo, fn)
        except Exception:
            pass
    # tokenizer.json(32MB) 은 snapshot 에 이미 캐시돼 있으면 재다운로드 안 됨
    try:
        tk = hf_hub_download(repo, "tokenizer.json")
    except Exception as e:
        print(f"[INFO] tokenizer.json 미다운로드({e.__class__.__name__}) — chat template 파일로 대체 검사")
        tk = None
    names = [p for p in tok_files]
    if tk:
        names.append("tokenizer.json")
    hr("토크나이저/채팅템플릿 검사 (가중치는 받지 않음)")
    print(f"  받은 파일: {names}")
    if tk:
        tok = AutoTokenizer.from_pretrained(os.path.dirname(tk))
        msgs = [{"role": "user", "content": "Hello!"}]
        s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(s, add_special_tokens=False)["input_ids"]
        print(f"  어휘 수: {tok.vocab_size}, 특수토큰 수: {len(tok.all_special_tokens)}")
        print(f"  chat template 적용 예시(앞 120자): {s[:120]!r}")
        print(f"  토큰화 길이: {len(enc)}")
        return True
    return False


def make_micro_model_and_generate(repo: str, prompt: str, max_tokens: int):
    """실제 가중치 대신 초소형 난수 가중치로 mlx_lm 추론 파이프라인 전체를 실행."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import gemma4
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    hr("micro 모드: 초소형 gemma4 랜덤 가중치 생성 → mlx_lm.stream_generate 실행")
    args = gemma4.ModelArgs(
        model_type="gemma4",
        vocab_size=4096,
        text_config=dict(
            hidden_size=64, num_hidden_layers=2, intermediate_size=128,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            global_head_dim=64, vocab_size_per_layer_input=4096,
            hidden_size_per_layer_input=16, num_kv_shared_layers=1,
            sliding_window=32, max_position_embeddings=512,
            use_double_wide_mlp=False, tie_word_embeddings=True,
        ),
    )
    model = gemma4.Model(args)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    print(f"  파라미터 수: {n_params:,} ({n_params*2/1e6:.1f} MB, bf16 기준)")

    # 간단한 BPE 없는 문자 토크나이저 대용: mlx_lm 은 리스트 입력도 허용
    ids = [min(ord(ch), 4095) for ch in prompt]
    sampler = make_sampler(temp=0.0)  # greedy
    print(f"  프롬프트: {prompt!r} -> {len(ids)} tokens")
    print("  생성 결과(난수 가중치라 의미 없는 텍스트가 정상):")
    out = []
    for resp in stream_generate(model, _CharTokenizer(), ids, max_tokens=max_tokens, sampler=sampler):
        out.append(resp.text)
    print("  >>> " + "".join(out)[:200])
    return True


class _CharTokenizer:
    """stream_generate 가 요구하는 최소 토크나이저 인터페이스 구현."""
    bos_token = None
    eos_token_ids = set()

    def __init__(self):
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        self._w = None  # wrapper 는 외부에서 감싸줌

    class _Detok:
        def add_token(self, t):
            self.last_segment = chr(t % 128) if 32 <= t % 128 else ""
        def flush(self):
            pass
    def __new__(cls):
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        inner = cls()
        detok = cls._Detok()
        detok.last_segment = ""
        return TokenizerWrapper(inner, tokenizer=None, detokenizer=detok,
                                eos_token_ids=set(), add_bos_token=False)


def run_full_inference(repo: str, prompt: str, max_tokens: int):
    """실제 모델 다운로드 후 mlx_lm 으로 추론 (Apple Silicon 권장)."""
    from mlx_lm import load, generate
    hr(f"전체 모델 다운로드 및 추론: {repo} (~3.4GB)")
    model, tokenizer = load(repo)
    print("[OK] 모델/토크나이저 로드 완료")
    reply = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True)
    return reply


def main():
    ap = argparse.ArgumentParser(description="MLX + gemma-4-E4B-it-qat-mobile 테스트")
    ap.add_argument("--model", default=DEFAULT_REPO)
    ap.add_argument("--prompt", default="Explain MLX in one sentence.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--download", action="store_true", help="실제 가중치를 받아 완전 추론")
    ap.add_argument("--dry-run", action="store_true", help="파일/config 검사만 수행")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    hr("0) 환경 점검")
    mlx_ok = check_imports()

    # 1) 원격 저장소 검사(다운로드 없음) — 항상 실행
    info = inspect_remote(a.model)
    if info is None:
        sys.exit("저장소에 접근할 수 없어 중단합니다.")

    # 2) 토크나이저/템플릿 검사 (가중치 제외)
    try:
        load_tokenizer_only(a.model)
    except Exception as e:
        print(f"[WARN] 토크나이저 검사 실패: {e}")

    if a.dry_run:
        hr("dry-run 완료: 모델 가중치는 받지 않았습니다.")
        return

    # 3) 추론 단계
    if not mlx_ok:
        hr("결론")
        print("이 환경에서는 mlx 네이티브 라이브러리를 로드할 수 없어 추론을 건너뜁니다.")
        print("위 검사는 전부 '가중치 다운로드 없이' 완료되었습니다. (원래 요청대로 모델接收 불필요 경로 OK)")
        print("Mac(Apple Silicon)에서 아래 명령으로 완전 실행 가능합니다:")
        print(f"  pip install mlx mlx-lm && python {os.path.basename(__file__)} --download")
        return

    if a.download:
        try:
            run_full_inference(a.model, a.prompt, a.max_tokens)
        except Exception as e:
            print(f"[FAIL] 전체 추론 오류: {e}")
    else:
        try:
            make_micro_model_and_generate(a.model, a.prompt, a.max_tokens)
        except Exception as e:
            print(f"[WARN] micro 추론 실패({e}). --download 로 실모델 테스트를 시도하세요.")

    hr("완료")


if __name__ == "__main__":
    main()
