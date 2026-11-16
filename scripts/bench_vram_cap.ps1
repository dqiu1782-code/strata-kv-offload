# Speed matrix on one large GPU: simulate smaller cards with STRATA_Q36_VRAM_CAP and compare
#   A = block selection, K/V in VRAM
#   C = block selection, K/V in pinned host memory + 256-slot GPU pool (offload)
# Each run: strata-q36 --bench-pp <prompt> --bench-tg 64 (random tokens, 3 runs inside the engine).
# The VRAM peak is sampled with nvidia-smi while the engine runs (the cap is soft, so check it).
param(
    [string]$Model = 'E:\llm\models\q36gguf\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf',
    [string]$Exe = 'E:\llm\strata_build\strata-q36.exe',
    [string]$Tmp = 'E:\llm\tmp',
    [int[]]$Cards = @(12, 16, 24, 32),
    [int[]]$Contexts = @(32768, 131072)
)
New-Item -ItemType Directory -Force $Tmp | Out-Null
$total = [int](nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
$idle = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
"GPU total $total MiB, idle used $idle MiB"

function Run($tag, $cardGiB, $ctx, $offload) {
    Get-ChildItem Env: | Where-Object { $_.Name -like 'STRATA_Q36_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
    $env:STRATA_Q36_SEL = '1'
    if ($offload) { $env:STRATA_Q36_KVHOST = '1'; $env:STRATA_Q36_KVPOOL = '256' }
    $env:STRATA_Q36_VRAM_CAP = "$cardGiB"
    $pp = if ($ctx -le 32768) { 30000 } else { 120000 }
    $out = "$Tmp\bench_$tag.out"; $err = "$Tmp\bench_$tag.err"
    $a = "--model `"$Model`" --ctx $ctx --bench-pp $pp --bench-tg 64"
    $t0 = Get-Date
    $p = Start-Process -FilePath $Exe -ArgumentList $a -PassThru -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError $err
    $peak = 0
    while (-not $p.HasExited) {
        $u = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        if ($u -gt $peak) { $peak = $u }
        Start-Sleep -Milliseconds 1500
    }
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    "=== $tag card=${cardGiB}GiB ctx=$ctx offload=$offload  peak $peak MiB of $total  wall ${secs}s  exit $($p.ExitCode) ==="
    Select-String -Path $out, $err -Pattern 'bench |hit rate|rror|VRAM free|pinned|hybrid experts: [0-9]+ of|every routed|out of memory' |
        ForEach-Object { $_.Line.Substring(0, [Math]::Min(210, $_.Line.Length)) } | Select-Object -First 14
    Start-Sleep 4
}

foreach ($ctx in $Contexts) {
    foreach ($card in $Cards) {
        Run "A${card}_$($ctx / 1024)k" $card $ctx $false
        Run "C${card}_$($ctx / 1024)k" $card $ctx $true
    }
}
Get-ChildItem Env: | Where-Object { $_.Name -like 'STRATA_Q36_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
'BENCH_DONE'
