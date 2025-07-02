import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.hnsw.*;
import org.apache.lucene.codecs.*;
import org.apache.lucene.codecs.lucene99.*;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Collectors;

// Custom IndexWriter that exposes the vector writer for graph injection
class CustomIndexWriter extends IndexWriter {
    private static Lucene99HnswVectorsWriter lastVectorWriter = null;
    
    public CustomIndexWriter(Directory d, IndexWriterConfig conf) throws IOException {
        super(d, conf);
    }
    
    public static Lucene99HnswVectorsWriter getLastVectorWriter() {
        return lastVectorWriter;
    }
    
    public static void setLastVectorWriter(Lucene99HnswVectorsWriter writer) {
        lastVectorWriter = writer;
        System.out.println("📝 Captured vector writer: " + writer.getClass().getSimpleName());
    }
}



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
            writer.println("GRAPH_LEVELS:1");  // Only level 0
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
            createSingleLevelConnections(writer, vectors);
        }
    }

    private static void createSingleLevelConnections(PrintWriter writer, List<float[]> vectors) {
        int maxConnections = 32; // More connections for single level
        int vectorCount = vectors.size();
        
        // Only create level 0 connections
        for (int nodeId = 0; nodeId < vectorCount; nodeId++) {
            List<Integer> connections = findNearestNeighbors(vectors, nodeId, 
                IntStream.range(0, vectorCount).boxed().collect(Collectors.toList()), maxConnections);
        
            writer.print("LEVEL_0_NODE_" + nodeId + ": [");
            for (int i = 0; i < connections.size(); i++) {
                writer.print(connections.get(i));
                if (i < connections.size() - 1) writer.print(", ");
            }
            writer.println("]");
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
        System.out.println("=== Starting Single-Level Graph Reconstruction ===");
        System.out.println("Vector count: " + data.vectorCount);
        System.out.println("Max connections: " + data.maxConnections);
        System.out.println("Expected levels: 1 (level 0 only)");
        
        // Create graph with fixed size to avoid extra levels
        OnHeapHnswGraph graph = new OnHeapHnswGraph(data.maxConnections, data.vectorCount);
        
        // Add all nodes to level 0 only
        System.out.println("Adding all " + data.vectorCount + " nodes to level 0...");
        int nodesAdded = 0;
        int nodesFailed = 0;
        
        for (int nodeId = 0; nodeId < data.vectorCount; nodeId++) {
            try {
                graph.addNode(0, nodeId);  // Add to level 0 only
                nodesAdded++;
                
                // Set entry node for the first node
                if (nodeId == 0) {
                    graph.trySetNewEntryNode(0, 0);
                }
            } catch (Exception e) {
                System.err.println("Failed to add node " + nodeId + " to level 0: " + e.getMessage());
                nodesFailed++;
            }
        }
        
        System.out.println("Nodes added: " + nodesAdded + ", failed: " + nodesFailed);
        System.out.println("Graph now has " + graph.numLevels() + " levels and " + graph.size() + " nodes");
        
        // Add connections for level 0
        int connectionsAdded = 0;
        int connectionsFailed = 0;
        
        Map<Integer, List<Integer>> level0Connections = data.connections.get(0);
        if (level0Connections != null) {
            System.out.println("Adding connections for " + level0Connections.size() + " nodes at level 0...");
            
            for (Map.Entry<Integer, List<Integer>> entry : level0Connections.entrySet()) {
                int nodeId = entry.getKey();
                List<Integer> neighbors = entry.getValue();
                
                if (neighbors.isEmpty()) continue;
                
                try {
                    NeighborArray neighborArray = graph.getNeighbors(0, nodeId);
                    int beforeSize = neighborArray.size();
                    
                    // Use the proper API method
                    addNeighborsToArrayRobust(neighborArray, neighbors);
                    // Or use this for actual scores: addNeighborsWithScores(neighborArray, neighbors, data, nodeId);
                    
                    int afterSize = neighborArray.size();
                    int added = afterSize - beforeSize;
                    connectionsAdded += added;
                    
                    if (nodeId < 5) { // Debug first few nodes
                        System.out.println("Node " + nodeId + ": added " + added + "/" + neighbors.size() + " neighbors");
                    }
                    
                } catch (Exception e) {
                    System.err.println("Failed to add neighbors for node " + nodeId + ": " + e.getMessage());
                    connectionsFailed += neighbors.size();
                }
            }
        }
        
        System.out.println("Connections added: " + connectionsAdded + ", failed: " + connectionsFailed);
        
        // Validate the final graph
        validateSingleLevelGraph(graph, data);
        
        return graph;
    }

    private static void addNeighborsToArrayRobust(NeighborArray na, List<Integer> neighbors) {
        try {
            System.out.println("Adding " + neighbors.size() + " neighbors using NeighborArray.addOutOfOrder()");
        
            int initialSize = na.size();
            int added = 0;
        
            for (Integer neighbor : neighbors) {
                try {
                    // Use addOutOfOrder since we don't care about sorting during reconstruction
                    // Use a dummy score of 1.0f - in a real scenario you'd use actual similarity scores
                    na.addOutOfOrder(neighbor.intValue(), 1.0f);
                    added++;
                } catch (IllegalStateException e) {
                    // NeighborArray is full (reached maxSize)
                    System.err.println("NeighborArray is full, cannot add more neighbors. Added " + added + " out of " + neighbors.size());
                    break;
                } catch (Exception e) {
                    System.err.println("Failed to add neighbor " + neighbor + ": " + e.getMessage());
                    break;
                }
            }
        
            int finalSize = na.size();
            System.out.println("Successfully added " + added + " neighbors (size: " + initialSize + " -> " + finalSize + ")");
        
        } catch (Exception e) {
            System.err.println("Failed to add neighbors to NeighborArray: " + e.getMessage());
            e.printStackTrace();
        }
    }

    
    

    private static void createIndexWithReconstructedData(GraphData data, OnHeapHnswGraph graph) throws IOException {
        System.out.println("=== Creating Index with Pre-built Custom Graph ===");
        
        Path indexPath = Paths.get("complete-injected-index");
        if (java.nio.file.Files.exists(indexPath)) deleteDirectory(indexPath.toFile());
        Directory directory = FSDirectory.open(indexPath);
        IndexWriterConfig config = new IndexWriterConfig();
        
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            // Add first document to trigger FieldWriter creation
            Document firstDoc = new Document();
            firstDoc.add(new KnnFloatVectorField("vector", data.vectors[0], VectorSimilarityFunction.COSINE));
            firstDoc.add(new StringField("id", "injected_0", Field.Store.YES));
            firstDoc.add(new StringField("original_id", "0", Field.Store.YES));
            firstDoc.add(new StringField("source", "reconstructed", Field.Store.YES));
            writer.addDocument(firstDoc);
            
            // Now try to inject the custom graph using the registered writer
            System.out.println("🔧 Attempting to inject custom graph...");
            Lucene99HnswVectorsWriter vectorWriter = Lucene99HnswVectorsWriter.getCurrentWriter();
            if (vectorWriter != null) {
                System.out.println("📍 Found registered vector writer!");
                Lucene99HnswVectorsWriter.FieldWriter<?> fieldWriter = vectorWriter.getFieldWriter("vector");
                if (fieldWriter != null) {
                    fieldWriter.setCustomGraph(graph);
                    System.out.println("✅ Custom graph injected successfully!");
                    System.out.println("📊 Custom graph has " + graph.size() + " nodes and " + graph.numLevels() + " levels");
                } else {
                    System.out.println("❌ FieldWriter not found for 'vector' field");
                }
            } else {
                System.out.println("❌ No registered vector writer found");
            }
            
            // Add remaining documents
            for (int i = 1; i < data.vectorCount; i++) {
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

    private static int getNeighborArraySize(NeighborArray na) {
        return na.size(); // Use the public method!
    }

    private static void validateSingleLevelGraph(OnHeapHnswGraph graph, GraphData data) {
        System.out.println("=== Single-Level Graph Validation ===");
        try {
            System.out.println("Graph size: " + graph.size());
            System.out.println("Graph levels: " + graph.numLevels());
            System.out.println("Max connections: " + graph.maxConn());
        
            // Check level 0 nodes
            HnswGraph.NodesIterator nodes = graph.getNodesOnLevel(0);
            int nodeCount = 0;
            int totalNeighbors = 0;
            int nodesWithNeighbors = 0;
        
            while (nodes.hasNext()) {
                int nodeId = nodes.nextInt();
                nodeCount++;
            
                try {
                    NeighborArray neighbors = graph.getNeighbors(0, nodeId);
                    int neighborCount = getNeighborArraySize(neighbors);
                    totalNeighbors += neighborCount;
                
                    if (neighborCount > 0) {
                        nodesWithNeighbors++;
                    }
                
                    // Show details for first few nodes
                    if (nodeId < 5) {
                        System.out.println("Node " + nodeId + " has " + neighborCount + " neighbors");
                    }
                
                } catch (Exception e) {
                    System.err.println("Failed to get neighbors for node " + nodeId + ": " + e.getMessage());
                }
            }
        
            System.out.println("Level 0 has " + nodeCount + " nodes");
            System.out.println("Total neighbors: " + totalNeighbors);
            System.out.println("Nodes with neighbors: " + nodesWithNeighbors + "/" + nodeCount);
            System.out.println("Average neighbors per node: " + (nodeCount > 0 ? (double)totalNeighbors/nodeCount : 0));
        
            // Check if graph is connected
            if (nodesWithNeighbors > 0) {
                System.out.println("✅ Graph has connections");
            } else {
                System.out.println("❌ Graph has no connections");
            }
        
        } catch (Exception e) {
            System.err.println("Graph validation failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
