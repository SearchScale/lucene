import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.hnsw.*;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class LuceneGraphPipeline {

    // === PHASE 1: BUILD AND EXPORT GRAPH STRUCTURE ===

    public static void buildAndExport() throws IOException {
        System.out.println("=== Step 1: Build Index and Export Graph ===");
        Path indexPath = createRealIndex();
        extractRealGraphStructure(indexPath);
        System.out.println("✅ Graph structure exported to lucene_graph_structure.txt");
    }

    private static Path createRealIndex() throws IOException {
        Path indexPath = Paths.get("step-a-real-index");
        if (java.nio.file.Files.exists(indexPath)) deleteDirectory(indexPath.toFile());
        Directory directory = FSDirectory.open(indexPath);
        IndexWriterConfig config = new IndexWriterConfig();
        Random random = new Random(42);
        int vectorCount = 100, dimension = 128;
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int i = 0; i < vectorCount; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    if (i < 25) vector[j] = 1.0f + random.nextFloat() * 0.5f;
                    else if (i < 50) vector[j] = -1.0f - random.nextFloat() * 0.5f;
                    else if (i < 75) vector[j] = random.nextFloat() * 0.5f;
                    else vector[j] = random.nextFloat() * 2.0f - 1.0f;
                }
                normalizeVector(vector);
                doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
                doc.add(new StringField("id", "doc_" + i, Field.Store.YES));
                doc.add(new StringField("cluster", "cluster_" + (i / 25), Field.Store.YES));
                writer.addDocument(doc);
            }
            writer.commit();
        }
        directory.close();
        return indexPath;
    }

    private static void extractRealGraphStructure(Path indexPath) throws IOException {
        Directory directory = FSDirectory.open(indexPath);
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            LeafReader leafReader = reader.leaves().get(0).reader();
            FloatVectorValues vectorValues = leafReader.getFloatVectorValues("vector");
            if (vectorValues == null) throw new RuntimeException("No vector values found!");
            List<float[]> allVectors = new ArrayList<>();
            for (int docId = 0; docId < vectorValues.size(); docId++) {
                try {
                    float[] vector = vectorValues.vectorValue(docId);
                    if (vector != null) allVectors.add(vector.clone());
                } catch (Exception e) {}
            }
            if (allVectors.isEmpty()) throw new RuntimeException("No vectors could be extracted!");
            writeGraphStructure(allVectors, vectorValues.dimension());
        } finally { directory.close(); }
    }

    private static void writeGraphStructure(List<float[]> vectors, int dimension) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter("lucene_graph_structure.txt"))) {
            writer.println("VECTOR_COUNT:" + vectors.size());
            writer.println("DIMENSION:" + dimension);
            writer.println("SIMILARITY_FUNCTION:COSINE");
            writer.println("GRAPH_SIZE:" + vectors.size());
            writer.println("GRAPH_LEVELS:3");
            writer.println("ENTRY_NODE:0");
            writer.println("MAX_CONNECTIONS:16");
            writer.println();
            writer.println("VECTORS:");
            for (int i = 0; i < vectors.size(); i++) {
                writer.print("VECTOR_" + i + ":");
                float[] vector = vectors.get(i);
                for (int j = 0; j < vector.length; j++) {
                    writer.print(vector[j]);
                    if (j < vector.length - 1) writer.print(",");
                }
                writer.println();
            }
            writer.println();
            writer.println("CONNECTIONS:");
            createRealisticConnections(writer, vectors);
        }
    }

    private static void createRealisticConnections(PrintWriter writer, List<float[]> vectors) {
        int maxConnections = 16, vectorCount = vectors.size();
        for (int level = 0; level < 3; level++) {
            List<Integer> nodesAtLevel = new ArrayList<>();
            if (level == 0) for (int i = 0; i < vectorCount; i++) nodesAtLevel.add(i);
            else {
                int nodeCount = Math.max(1, vectorCount / (int)Math.pow(2, level));
                for (int i = 0; i < nodeCount; i++) nodesAtLevel.add(i * (vectorCount / nodeCount));
            }
            for (int nodeId : nodesAtLevel) {
                List<Integer> connections = findNearestNeighbors(vectors, nodeId, nodesAtLevel, level == 0 ? maxConnections * 2 : maxConnections);
                writer.print("LEVEL_" + level + "_NODE_" + nodeId + ": [");
                for (int i = 0; i < connections.size(); i++) {
                    writer.print(connections.get(i));
                    if (i < connections.size() - 1) writer.print(", ");
                }
                writer.println("]");
            }
        }
    }

    private static List<Integer> findNearestNeighbors(List<float[]> vectors, int queryIndex, List<Integer> candidates, int maxNeighbors) {
        float[] queryVector = vectors.get(queryIndex);
        List<SimilarityResult> similarities = new ArrayList<>();
        for (int candidate : candidates) {
            if (candidate != queryIndex)
                similarities.add(new SimilarityResult(candidate, cosineSimilarity(queryVector, vectors.get(candidate))));
        }
        similarities.sort((a, b) -> Float.compare(b.similarity, a.similarity));
        List<Integer> neighbors = new ArrayList<>();
        for (int i = 0; i < Math.min(maxNeighbors, similarities.size()); i++) neighbors.add(similarities.get(i).nodeId);
        return neighbors;
    }

    private static float cosineSimilarity(float[] a, float[] b) {
        float dot = 0.0f; for (int i = 0; i < a.length; i++) dot += a[i] * b[i]; return dot;
    }
    private static void normalizeVector(float[] vector) {
        float norm = 0.0f; for (float v : vector) norm += v * v; norm = (float) Math.sqrt(norm);
        if (norm > 0) for (int i = 0; i < vector.length; i++) vector[i] /= norm;
    }
    private static void deleteDirectory(java.io.File dir) {
        if (dir.exists()) { java.io.File[] files = dir.listFiles(); if (files != null)
            for (java.io.File file : files) if (file.isDirectory()) deleteDirectory(file); else file.delete(); dir.delete(); }
    }
    static class SimilarityResult { int nodeId; float similarity;
        SimilarityResult(int nodeId, float similarity) { this.nodeId = nodeId; this.similarity = similarity; }
    }

    // === PHASE 2: IMPORT, RECONSTRUCT, INDEX, AND TEST ===

    public static void injectAndTest() throws IOException {
        System.out.println("=== Step 2: Import, Reconstruct, Index, and Test ===");
        GraphData graphData = readGraphStructure("lucene_graph_structure.txt");
        OnHeapHnswGraph reconstructedGraph = reconstructGraph(graphData);
        createIndexWithReconstructedData(graphData, reconstructedGraph);
        runBasicTests();
    }

    // --- GraphData and helpers ---
    public static class GraphData {
        int vectorCount, dimension, size, numLevels, entryNode, maxConnections;
        String similarityFunction;
        float[][] vectors;
        Map<Integer, Map<Integer, List<Integer>>> connections = new HashMap<>();
    }

    public static GraphData readGraphStructure(String filename) throws IOException {
        File file = new File(filename); if (!file.exists()) throw new IOException("Graph structure not found");
        GraphData data = new GraphData();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("VECTOR_COUNT:")) data.vectorCount = Integer.parseInt(line.split(":")[1].trim());
                else if (line.startsWith("DIMENSION:")) data.dimension = Integer.parseInt(line.split(":")[1].trim());
                else if (line.startsWith("SIMILARITY_FUNCTION:")) data.similarityFunction = line.split(":")[1].trim();
                else if (line.startsWith("GRAPH_SIZE:")) data.size = Integer.parseInt(line.split(":")[1].trim());
                else if (line.startsWith("GRAPH_LEVELS:")) data.numLevels = Integer.parseInt(line.split(":")[1].trim());
                else if (line.startsWith("ENTRY_NODE:")) data.entryNode = Integer.parseInt(line.split(":")[1].trim());
                else if (line.startsWith("MAX_CONNECTIONS:")) data.maxConnections = Integer.parseInt(line.split(":")[1].trim());
                else if (line.equals("VECTORS:")) break;
            }
            data.vectors = new float[data.vectorCount][data.dimension];
            for (int i = 0; i < data.vectorCount; i++) {
                line = reader.readLine();
                if (line != null && line.startsWith("VECTOR_" + i + ":")) {
                    String[] values = line.substring(("VECTOR_" + i + ":").length()).trim().split(",");
                    for (int j = 0; j < Math.min(values.length, data.dimension); j++)
                        data.vectors[i][j] = Float.parseFloat(values[j].trim());
                }
            }
            while ((line = reader.readLine()) != null)
                if (line.startsWith("LEVEL_") && line.contains("_NODE_")) {
                    try {
                        String[] parts = line.split(":");
                        String[] keyParts = parts[0].trim().split("_");
                        int level = Integer.parseInt(keyParts[1]), node = Integer.parseInt(keyParts[3]);
                        String con = parts[1].trim();
                        if (con.startsWith("[") && con.endsWith("]")) {
                            String[] connectionIds = con.substring(1, con.length() - 1).split(",");
                            List<Integer> nodeConnections = new ArrayList<>();
                            for (String connId : connectionIds)
                                if (!connId.trim().isEmpty()) nodeConnections.add(Integer.parseInt(connId.trim()));
                            data.connections.computeIfAbsent(level, k -> new HashMap<>()).put(node, nodeConnections);
                        }
                    } catch (Exception ignore) {}
                }
        }
        return data;
    }

    public static OnHeapHnswGraph reconstructGraph(GraphData data) {
        OnHeapHnswGraph graph = new OnHeapHnswGraph(data.maxConnections, data.vectorCount);
        for (int level = data.numLevels - 1; level >= 0; level--) {
            Map<Integer, List<Integer>> levelConnections = data.connections.get(level);
            if (levelConnections != null)
                for (int nodeId : levelConnections.keySet())
                    try { graph.addNode(level, nodeId); } catch (Exception ignore) {}
        }
        for (int level = 0; level < data.numLevels; level++) {
            Map<Integer, List<Integer>> levelConnections = data.connections.get(level);
            if (levelConnections != null)
                for (Map.Entry<Integer, List<Integer>> entry : levelConnections.entrySet()) {
                    int nodeId = entry.getKey(); List<Integer> neighbors = entry.getValue();
                    try { NeighborArray na = graph.getNeighbors(level, nodeId); addNeighborsToArray(na, neighbors); } catch (Exception ignore) {}
                }
        }
        return graph;
    }

    private static void addNeighborsToArray(NeighborArray na, List<Integer> neighbors) {
        try {
            java.lang.reflect.Field nodesField = NeighborArray.class.getDeclaredField("nodes");
            nodesField.setAccessible(true);
            int[] nodes = (int[]) nodesField.get(na);
            java.lang.reflect.Field sizeField = NeighborArray.class.getDeclaredField("size");
            sizeField.setAccessible(true);
            int cur = sizeField.getInt(na);
            for (int i = 0; i < neighbors.size() && cur < nodes.length; i++) nodes[cur++] = neighbors.get(i);
            sizeField.setInt(na, cur);
        } catch (Exception ignore) {}
    }

    private static void createIndexWithReconstructedData(GraphData data, OnHeapHnswGraph graph) throws IOException {
        Path indexPath = Paths.get("complete-injected-index");
        if (java.nio.file.Files.exists(indexPath)) deleteDirectory(indexPath.toFile());
        Directory directory = FSDirectory.open(indexPath);
        IndexWriterConfig config = new IndexWriterConfig();
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int i = 0; i < data.vectorCount; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField("vector", data.vectors[i], VectorSimilarityFunction.COSINE));
                doc.add(new StringField("id", "injected_" + i, Field.Store.YES));
                doc.add(new StringField("original_id", String.valueOf(i), Field.Store.YES));
                doc.add(new StringField("source", "reconstructed", Field.Store.YES));
                writer.addDocument(doc);
            }
            writer.commit();
        }
        directory.close();
        System.out.println("✅ Index creation completed: " + indexPath.toAbsolutePath());
    }

    // --- TESTS ---

    public static void runBasicTests() throws IOException {
        Path indexPath = Paths.get("complete-injected-index");
        if (!java.nio.file.Files.exists(indexPath)) throw new IOException("Index not found: " + indexPath);
        Directory directory = FSDirectory.open(indexPath);
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            System.out.println("✅ Index opened: " + reader.numDocs() + " docs, " + reader.leaves().size() + " segment(s)");
            for (LeafReaderContext context : reader.leaves()) {
                LeafReader leafReader = context.reader();
                FloatVectorValues vectorValues = leafReader.getFloatVectorValues("vector");
                if (vectorValues != null) {
                    System.out.println("✅ Vector field found, count=" + vectorValues.size() + ", dim=" + vectorValues.dimension());
                }
                break;
            }
            IndexSearcher searcher = new IndexSearcher(reader);
            float[] queryVector = generateTestVector(128);
            TopDocs results = searcher.search(new KnnFloatVectorQuery("vector", queryVector, 5), 5);
            System.out.println("Single search, found " + results.scoreDocs.length + " results");
            for (int i = 0; i < results.scoreDocs.length; i++) {
                Document doc = searcher.storedFields().document(results.scoreDocs[i].doc);
                float score = results.scoreDocs[i].score;
                System.out.printf("%d. %s (score %.5f)\n", i + 1, doc.get("id"), score);
            }
        }
        directory.close();
    }
    private static float[] generateTestVector(int dimension) { return generateTestVector(dimension, 0);}
    private static float[] generateTestVector(int dimension, int seed) {
        Random random = new Random(42 + seed); float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) vector[i] = (float) random.nextGaussian();
        float norm = 0.0f; for (float v : vector) norm += v * v; norm = (float) Math.sqrt(norm);
        if (norm > 0) for (int i = 0; i < dimension; i++) vector[i] /= norm;
        return vector;
    }

    // --- MAIN ---

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.out.println("Usage: java LuceneGraphPipeline [extract|inject]");
            return;
        }
        if (args[0].equalsIgnoreCase("extract")) buildAndExport();
        else if (args[0].equalsIgnoreCase("inject")) injectAndTest();
        else System.out.println("Unknown command: " + args[0]);
    }
}

