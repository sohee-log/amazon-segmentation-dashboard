# 팀원 작업 공간 (members/)

각자 **자기 폴더 안에서만** 작업하면 충돌(conflict) 없이 협업할 수 있어요.

## 폴더 규칙

| 폴더 | 담당 | 대시보드 탭 |
|------|------|-------------|
| `sohee/` | sohee | 1. Product Segmentation |
| `member2/` | (팀원 B) | 2. Teammate Analysis |
| `member3/` | (팀원 C) | 3. Teammate Analysis |

> `member2`, `member3` 폴더는 각자 **자기 이름(영문)** 으로 바꿔도 됩니다.
> 이름을 바꿨다면 팀 채팅에 공유해서 서로 헷갈리지 않게 해주세요.

## 처음 시작하는 법

```bash
# 1. 레포 복제 (최초 1회)
git clone https://github.com/sohee-log/amazon-segmentation-dashboard.git
cd amazon-segmentation-dashboard

# 2. 최신 상태로 맞추기 (작업 시작 전 항상)
git pull origin main

# 3. 자기 폴더에서 작업 후 올리기
git add members/member2/       # 자기 폴더만 add
git commit -m "add: (이름) 분석 초안"
git push origin main
```

## 충돌을 피하는 3가지 습관

1. **작업 시작 전 `git pull`** — 남이 올린 최신 내용을 먼저 받는다.
2. **자기 폴더만 수정** — `app.py` 같은 공용 파일은 함께 상의 후 수정.
3. **자주 작게 commit·push** — 오래 쌓아두면 충돌 확률이 커진다.

## 무엇을 넣나요?

각 폴더에 자유롭게:

- 분석 노트북(`.ipynb`) / 스크립트(`.py`)
- 결과 이미지, 요약 CSV
- 자기 파트 설명 `README.md`

대시보드 탭에 연결하고 싶으면 `app.py`의 `render_teammate_tab()` 부분을
같이 상의해서 채우면 됩니다.
