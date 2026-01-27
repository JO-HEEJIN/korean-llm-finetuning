# Korean LLM Fine-tuning Project - Todo

## Problem
이전 커밋 메시지에 Claude 세션 URL이 포함되어 있어 규칙 위반. 수정 필요.

## Plan

### To-Do List
- [x] Git 사용자 설정 변경 (JO-HEEJIN, midmost44@gmail.com)
- [x] 커밋 메시지에서 Claude URL 제거 (amend)
- [x] 원격 저장소에 강제 푸시

## Review

### 변경 사항 요약
1. Git 사용자 설정을 JO-HEEJIN (midmost44@gmail.com)으로 변경
2. 커밋 메시지에서 Claude 세션 URL 제거
3. 커밋 Author를 Claude에서 JO-HEEJIN으로 변경 (--reset-author)
4. 원격 저장소에 강제 푸시하여 이전 커밋 덮어쓰기

### 사용한 명령어
- `git config user.name` / `git config user.email`
- `git commit --amend --reset-author`
- `git push --force`
