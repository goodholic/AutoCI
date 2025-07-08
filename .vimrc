" Vim의 현재 작업 디렉토리를 이 파일이 있는 곳으로 변경
lcd %:p:h

" 알림: AutoCi 프로젝트 설정이 로드되었습니다.
echom "Loaded AutoCi project settings. Working directory changed to " . getcwd()

