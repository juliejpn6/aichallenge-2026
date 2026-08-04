# Claude Code環境 運用ガイド(1ページ)

このリポジトリのClaude Code設定(`CLAUDE.md` / `.claude/skills/` /
`.claude/commands/` / `.claude/hooks-draft/`)を人間側からどう使うかのガイド。

## `#`メッセージによる即時メモリ追記

方針決定をした瞬間、忘れないうちに`#`で始まるメッセージを打つと、
Claude Codeの永続メモリに直接記録される(セッションをまたいで保持される保険)。
例:

```
# Q[e_psi]は1000000で確定、2.5e6以上はP→Rバグ修正まで再評価しない
```

これは`design_docs`やCLAUDE.mdへの反映とは別経路の「即時メモ」であり、
形式立った記録が欲しい場合は`/log-decision <内容>`を使う(下記)。両方使っても
問題ない(`#`は保険、`/log-decision`は正式な記録)。

**注意**: `#`メモはCLAUDE.md §0のルールにも`/audit-env`の監査対象にも入らない、
完全に別経路の記録である。つまり`#`で打った内容とCLAUDE.mdが将来矛盾しても、
`/audit-env`はそれを検出できない。`#`だけで済ませた決定は必ず後日
`/log-decision`で正式記録へ昇格させること——`#`はあくまで一時的な保険であり、
正式な記録の代わりにはならない。

## `/log-decision` — 方針決定の正式記録

方針決定をした際、`/log-decision <決定内容>`を打つと、design_docsの新節と
CLAUDE.mdの該当セクションへの追記下書きを作り、diffで提示する。**内容を確認
してから承認する**(自動コミットはされない)。

## `/status` — 現状確認

作業を始める前や、しばらく間を置いた後に`/status`を打つと、config.yamlの
レース値維持キーに未コミット差分が無いか、直近のdesign_docs節、直近コミットの
傾向を一画面で確認できる。

## `/restore-race-config` — 実験後の復元

パラメータ実験でconfig.yamlを変更した後、レース値へ戻す際に使う。差分を提示し、
承認後に復元する(コミットは別途)。

## `/audit-env` — 定期監査の実行タイミング

CLAUDE.mdの内容(特に§3禁止リスト)が実態とズレていないかを機械的に洗い出す
コマンド。以下のタイミングでの実行を推奨する:

- **フェーズ節目**(Phase1出口ゲート判定時、v_max段階を引き上げる時等)
- **内部締切前後**(2026-08-25の凍結移行時)
- **外部環境の変更を適用した直後**(AWSIMのアップデート等。運営側のsteer rate
  変更がプラント特性を変えた実績があり、環境が変わった際は初日測定キット
  [`step_response_test.py`]と並んで、CLAUDE.mdの禁止リスト・環境情報が
  新環境でも正しいかの監査が必要になる)
- 何か「あれ、これ前に禁止したはずでは?」と違和感を覚えた時(いつでも歓迎)

読み取り専用の監査であり、見つかったズレの反映は人間の承認を得てから別途行う。

## フック承認の手順

`.claude/hooks-draft/`配下のフック・permissions提案は**まだ有効化されていない**。
有効化したい場合は`.claude/hooks-draft/README.md`の手順に従うこと。要点:

1. `hooks.settings.json`の中身を`.claude/settings.json`または
   `.claude/settings.local.json`の`hooks`キーへ手動でマージする。
2. `permissions-proposal.json`から承認した項目だけを
   `permissions.allow`/`permissions.ask`へ追記する。
3. 新しいセッションを開始し(フックはセッション開始時に読み込まれる)、
   CLAUDE.mdを軽く編集してみる。**これが`permissionDecision: "ask"`の
   動作確認を兼ねる**——確認ダイアログが出れば対応している。何も起きず
   素通りする場合は未対応なので、`claude_md_edit_guard.sh`を単純な
   `exit 2`(無条件ブロック+stderr表示)方式へ書き換える
   (`.claude/hooks-draft/README.md`の該当箇所参照)。

承認後は、以下のテンプレートでClaude Codeへ結果を報告すると記録が残しやすい
(コピーして値を埋めるだけで使える):

```
[環境設定 結果報告] YYYY-MM-DD

承認/適用した項目:
- [ ] Stopフック(scripts/check_race_config.sh)を有効化
- [ ] PreToolUseフック(claude_md_edit_guard.sh、CLAUDE.md・安全判定系ファイル編集時の確認)を有効化
- [ ] permissions-proposal.jsonのproposed_allowから追記した項目: (列挙)
- [ ] permissions-proposal.jsonのproposed_askから追記した項目: (列挙)
- [ ] その他: (自由記述)

見送った項目・条件を変えて適用した項目:
(自由記述。理由も一言添えてください)

Claudeへの依頼:
- 上記をdocs/claude-env-guide.mdの状態と整合するよう確認してください
- /status を実行して整合性を確認してください
- 動作確認(CLAUDE.md編集時にask確認が出るか等)の結果も併せて報告してください
```

## CLAUDE.md更新diffのレビュー観点

CLAUDE.mdの§0ルールにより、Claude Codeはタスク中に確定結論が変わった場合
CLAUDE.mdを自己更新する。そのdiffをレビューする際は以下を確認する:

- **根拠(design_docsの節番号)が併記されているか**。無ければ差し戻す(§0違反)。
- §3(禁止リスト)への追加なら、既存の関連項目と矛盾していないか(古い項目が
  正しく削除/更新されているか)。
- §1/§2の変更なら、値そのものではなく「規律・型」の変更に留まっているか
  (実測値・現在のパラメータが紛れ込んでいないか——薄い索引の原則)。
- 変更範囲が今回のタスクの根拠から逸脱して大きすぎないか(便乗した無関係な
  書き換えが無いか)。
