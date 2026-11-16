# Windows build of Strata's strata-q36 target (the paths are the test machine's; change them for yours).
# Needs: Visual Studio 2022 Build Tools (vcvars64.bat), CUDA 13.0, CMake + Ninja, and a llama.cpp checkout that
# Strata's CMake would otherwise download (FETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP).
param(
    [string]$Src = 'E:\llm\strata',
    [string]$Build = 'E:\llm\strata_build',
    [string]$Vcvars = 'E:\llm\vs2022\VC\Auxiliary\Build\vcvars64.bat',
    [string]$Cuda = 'E:\llm\cuda\v13.0',
    [string]$LlamaCpp = 'E:\llm\llamacpp',
    [string]$Arch = '120'   # RTX 50 series; 89 = RTX 40, 86 = RTX 30
)
$ErrorActionPreference = 'Continue'
cmd /c "`"$Vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:CUDA_PATH = $Cuda; $env:CUDA_HOME = $Cuda
$env:PATH = "$Cuda\bin;$Cuda\bin\x64;$env:PATH"
Remove-Item -Recurse -Force $Build -ErrorAction SilentlyContinue
cmake -S $Src -B $Build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$Arch `
    -DCUDAToolkit_ROOT="$Cuda" -DCMAKE_CUDA_COMPILER="$Cuda\bin\nvcc.exe" `
    -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=$LlamaCpp -DFETCHCONTENT_FULLY_DISCONNECTED=ON
"CONFIGURE_EXIT $LASTEXITCODE"
if ($LASTEXITCODE -eq 0) {
    cmake --build $Build --target strata-q36
    "BUILD_EXIT $LASTEXITCODE"
}
