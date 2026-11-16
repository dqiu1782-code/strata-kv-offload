# Quality runs on the public corpus (Strata source), K / V offloaded to pinned host memory + 256-slot GPU pool.
param([string]$Py = 'E:\llm\envs\kvhybrid\python.exe', [string]$Dir = 'E:\llm\kvsel_release\scripts', [string]$Log = 'E:\llm')
$env:STRATA_Q36_KVHOST = '1'
$env:STRATA_Q36_KVPOOL = '256'
foreach ($c in @(@{ctx = 60000; steps = 160}, @{ctx = 120000; steps = 96})) {
    $env:SEL_CTX = "$($c.ctx)"
    $env:SEL_STEPS = "$($c.steps)"
    & $Py "$Dir\sel_eval.py" *> "$Log\sel_eval_public$($c.ctx / 1000)k.log"
}
'PUBLIC_EVAL_DONE' | Out-File "$Log\sel_eval_public_done.txt"
