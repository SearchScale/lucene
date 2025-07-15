
## Compilation and Execution Steps

### Step 1: Create Build Directory and Compile Lucene Core

```bash
# Create build directory
mkdir -p /tmp/lucene-build

# Copy Lucene resources (needed for SPI/Codec loading)
cp -r lucene/core/src/resources/* /tmp/lucene-build/

# Compile Lucene core classes
javac -cp "lucene/core/src/java:lucene/core/src/java24" -d /tmp/lucene-build \
  $(find lucene/core/src/java -name "*.java" | grep -E "(codecs|util|search|document|index|store|analysis)" | head -500)
```

### Step 2: Compile LuceneGraphPipeline

```bash
javac -cp "/tmp/lucene-build:lucene/core/src/java:lucene/core/src/java24" -d /tmp/lucene-build LuceneGraphPipeline.java
```

### Step 3: Run the Program

```bash
java -cp "/tmp/lucene-build" org.apache.lucene.demo.LuceneGraphPipeline cagra_graph_data26.txt
```

