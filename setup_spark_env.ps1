# Setup Spark, Hadoop, and Java Environment Variables
# Run as Administrator: Right-click PowerShell -> Run as Administrator
# Then execute: Set-ExecutionPolicy Bypass -Scope Process -Force; .\setup_spark_env.ps1

Write-Host "Setting up Spark 4.0.1 + Hadoop 3.3.5 + Java 17 environment..." -ForegroundColor Green

# Set JAVA_HOME to JDK 17
$JAVA_HOME = "C:\Program Files\Java\jdk-17"
[System.Environment]::SetEnvironmentVariable("JAVA_HOME", $JAVA_HOME, [System.EnvironmentVariableTarget]::Machine)
Write-Host "✓ JAVA_HOME set to: $JAVA_HOME" -ForegroundColor Cyan

# Set HADOOP_HOME
$HADOOP_HOME = "C:\hadoop"
[System.Environment]::SetEnvironmentVariable("HADOOP_HOME", $HADOOP_HOME, [System.EnvironmentVariableTarget]::Machine)
Write-Host "✓ HADOOP_HOME set to: $HADOOP_HOME" -ForegroundColor Cyan

# Set SPARK_HOME (update this path to where you installed Spark 4.0.1)
$SPARK_HOME = "C:\spark"  # Change this if Spark is installed elsewhere
if (Test-Path $SPARK_HOME) {
    [System.Environment]::SetEnvironmentVariable("SPARK_HOME", $SPARK_HOME, [System.EnvironmentVariableTarget]::Machine)
    Write-Host "✓ SPARK_HOME set to: $SPARK_HOME" -ForegroundColor Cyan
} else {
    Write-Host "⚠ SPARK_HOME directory not found at $SPARK_HOME" -ForegroundColor Yellow
    Write-Host "  Please install Spark 4.0.1 or update the path in this script" -ForegroundColor Yellow
}

# Update PATH
$currentPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine)

# Remove old Java paths
$pathArray = $currentPath -split ';' | Where-Object { $_ -notmatch 'Java\\jdk-20' }

# Add new paths if not already present
$pathsToAdd = @(
    "$JAVA_HOME\bin",
    "$HADOOP_HOME\bin",
    "$SPARK_HOME\bin"
)

foreach ($path in $pathsToAdd) {
    if ($pathArray -notcontains $path -and (Test-Path $path)) {
        $pathArray += $path
        Write-Host "✓ Added to PATH: $path" -ForegroundColor Cyan
    }
}

$newPath = ($pathArray | Where-Object { $_ -ne "" }) -join ';'
[System.Environment]::SetEnvironmentVariable("Path", $newPath, [System.EnvironmentVariableTarget]::Machine)

Write-Host "`n✅ Environment variables configured successfully!" -ForegroundColor Green
Write-Host "`n⚠ IMPORTANT: You must RESTART your terminal/IDE for changes to take effect!" -ForegroundColor Yellow
Write-Host "`nTo verify, run these commands in a NEW terminal:" -ForegroundColor White
Write-Host "  java -version    (should show Java 17)" -ForegroundColor Gray
Write-Host "  echo `$env:HADOOP_HOME    (should show C:\hadoop)" -ForegroundColor Gray
Write-Host "  echo `$env:SPARK_HOME     (should show C:\spark)" -ForegroundColor Gray
