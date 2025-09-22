# Push Failure Remediation Notes (2025-09-22)

## 背景問題
- 在本地新增 `XSUM` 資料集與相關輸出後，意外地把大型檔案一同提交，導致遠端拒絕推送。
- 清理推送內容後再次嘗試，GitHub Push Protection 偵測到 `run.sh` 中硬編碼的 OpenAI API key，推送仍被阻擋。

## 解決步驟
1. 使用 `git reset --hard HEAD~1` 移除含大型檔案的提交，之後從 `git reflog` 取回需要的程式與資料（保留於本地並忽略版本控制）。
2. 將 `run.sh` 中直接暴露的金鑰改為從環境變數讀取，避免秘密進入版本控制。
3. 加上可匯入的本地設定檔（不納入 Git）機制，讓指令腳本在執行時自動載入 API 金鑰。
4. 重新修訂提交並推送，確認遠端更新成功。

## 後續建議
- 將實際金鑰放在 `config/api_keys.sh`（已列入 `.gitignore`），或設定 `RUN_CONFIG_FILE` 指向其他路徑；必要時可以為不同供應商（如 OpenAI、DeepSeek）設定相對應的金鑰。
- 如金鑰曾經暴露，務必在供應商後台註銷舊金鑰並建立新的，並檢查其他分支或備份是否含有同樣資訊。

