package org.apache.lucene.demo;

import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.hnsw.*;
import org.apache.lucene.codecs.lucene99.*;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * CAGRA to Lucene HNSW Graph Injection Pipeline
 * 
 * This pipeline demonstrates how to import CAGRA graph structures into Lucene's HNSW implementation
 */
public class LuceneGraphPipeline {
    
    // === MAIN ENTRY POINT ===
    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.out.println("Usage: java LuceneGraphPipeline <cagra_graph_file>");
            return;
        }
        importCagraGraphAndIndex(args[0]);
    }
    
    // === CORE PIPELINE ===
    public static void importCagraGraphAndIndex(String cagraFile) throws IOException {
        System.out.println("=== Importing CAGRA Graph and Indexing ===");
        GraphData graphData = readCagraGraphFile(cagraFile);
        
        // Add validation here
        validateGraphStructure(graphData);
        
        OnHeapHnswGraph reconstructedGraph = reconstructGraph(graphData);
        createIndexWithReconstructedData(graphData, reconstructedGraph);
        runQueryVectorTests(graphData);
    }
    
    // === DATA STRUCTURE ===
    public static class GraphData {
        public int vectorCount, dimension, size, numLevels, entryNode, maxConnections;
        public String similarityFunction;
        public float[][] vectors;        // Dataset vectors
        public float[][] queryVectors;   // Query vectors for testing
        public int queryCount;           // Number of query vectors
        public Map<Integer, Map<Integer, List<Integer>>> connections = new HashMap<>();
    }
    
    /**
     * Parse CAGRA graph file containing adjacency lists and vector data
     */
    public static GraphData readCagraGraphFile(String filename) throws IOException {
        File file = new File(filename); 
        if (!file.exists()) throw new IOException("CAGRA graph file not found: " + filename);
        
        GraphData data = new GraphData();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            boolean inAdjacency = false, inDatasetVectors = false, inQueryVectors = false;
            List<String> adjacencyLines = new ArrayList<>();
            List<String> datasetVectorLines = new ArrayList<>();
            List<String> queryVectorLines = new ArrayList<>();
            
            // Phase 1: Collect sections and detect vector count
            data.vectorCount = -1;
            data.queryCount = -1;
            
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                
                // Parse dataset vector count
                if (line.startsWith("Dataset vectors count: ")) {
                    try {
                        String countStr = line.substring("Dataset vectors count: ".length()).trim();
                        data.vectorCount = Integer.parseInt(countStr);
                        System.out.println("📊 Detected dataset vector count: " + data.vectorCount);
                    } catch (NumberFormatException e) {
                        System.err.println("⚠️ Could not parse dataset vector count from: " + line);
                    }
                    continue;
                }
                
                // Parse query vector count
                if (line.startsWith("Query vectors count: ")) {
                    try {
                        String countStr = line.substring("Query vectors count: ".length()).trim();
                        data.queryCount = Integer.parseInt(countStr);
                        System.out.println("📊 Detected query vector count: " + data.queryCount);
                    } catch (NumberFormatException e) {
                        System.err.println("⚠️ Could not parse query vector count from: " + line);
                    }
                    continue;
                }
                
                // Parse vector dimension
                if (line.startsWith("Vector dimensions: ")) {
                    try {
                        String dimStr = line.substring("Vector dimensions: ".length()).trim();
                        data.dimension = Integer.parseInt(dimStr);
                        System.out.println("📊 Detected vector dimension: " + data.dimension);
                    } catch (NumberFormatException e) {
                        System.err.println("⚠️ Could not parse vector dimension from: " + line);
                    }
                    continue;
                }
                
                // Try to parse vector count from adjacency list header (fallback)
                if (line.startsWith("Complete adjacency list for all ") && line.endsWith(" nodes:")) {
                    String headerText = line.substring("Complete adjacency list for all ".length(), 
                                                       line.length() - " nodes:".length()).trim();
                    try {
                        int adjCount = Integer.parseInt(headerText);
                        if (data.vectorCount == -1) {
                            data.vectorCount = adjCount;
                            System.out.println("📊 Detected vector count from adjacency header: " + data.vectorCount);
                        } else if (data.vectorCount != adjCount) {
                            System.out.println("⚠️ Adjacency count (" + adjCount + ") differs from dataset count (" + data.vectorCount + ")");
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("⚠️ Could not parse vector count from header: " + line);
                    }
                    inAdjacency = true; 
                    inDatasetVectors = false;
                    inQueryVectors = false;
                    continue;
                }
                
                // Alternative: look for dataset size
                if (line.startsWith("Dataset Size: ")) {
                    try {
                        String sizeStr = line.substring("Dataset Size: ".length()).trim();
                        int datasetSize = Integer.parseInt(sizeStr);
                        if (data.vectorCount == -1) {
                            data.vectorCount = datasetSize;
                            System.out.println("📊 Detected vector count from dataset size: " + data.vectorCount);
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("⚠️ Could not parse dataset size from: " + line);
                    }
                    continue;
                }
                
                if (line.startsWith("=== ALL ") && line.contains("DATASET VECTORS")) {
                    inAdjacency = false; 
                    inDatasetVectors = true;
                    inQueryVectors = false;
                    System.out.println("📊 Found dataset vectors section");
                    continue;
                }
                
                if (line.startsWith("=== ALL ") && line.contains("QUERY VECTORS")) {
                    inAdjacency = false; 
                    inDatasetVectors = false;
                    inQueryVectors = true;
                    System.out.println("📊 Found query vectors section");
                    continue;
                }
                
                if ((inAdjacency || inDatasetVectors || inQueryVectors) && line.startsWith("===")) {
                    inAdjacency = false;
                    inDatasetVectors = false;
                    inQueryVectors = false;
                    continue;
                }
                
                if (inAdjacency && !line.isEmpty()) {
                    adjacencyLines.add(line);
                }
                if (inDatasetVectors && !line.isEmpty()) {
                    datasetVectorLines.add(line);
                }
                if (inQueryVectors && !line.isEmpty()) {
                    queryVectorLines.add(line);
                }
            }
            
            // Set defaults if not found
            if (data.vectorCount == -1) {
                data.vectorCount = 2297; // Fallback
                System.out.println("⚠️ Using fallback vector count: " + data.vectorCount);
            }
            
            if (data.queryCount == -1) {
                data.queryCount = 8; // Fallback
                System.out.println("⚠️ Using fallback query count: " + data.queryCount);
            }
            
            if (data.dimension == -1) {
                data.dimension = 201; // Fallback
                System.out.println("⚠️ Using fallback dimension: " + data.dimension);
            }
            
            // Phase 2: Parse adjacency list
            data.similarityFunction = "COSINE";
            data.connections = new HashMap<>();
            Map<Integer, List<Integer>> level0 = new HashMap<>();
            
            for (String adj : adjacencyLines) {
                if (!adj.startsWith("Node ") || !adj.contains(":")) continue;
                
                try {
                    int colonIdx = adj.indexOf(":");
                    String nodeIdStr = adj.substring(5, colonIdx).trim();
                    
                    if (!nodeIdStr.matches("\\d+")) continue;
                    
                    int nodeId = Integer.parseInt(nodeIdStr);
                    String rest = adj.substring(colonIdx + 1).trim();
                    if (rest.startsWith("[")) rest = rest.substring(1);
                    if (rest.endsWith("]")) rest = rest.substring(0, rest.length() - 1);
                    
                    List<Integer> neighbors = new ArrayList<>();
                    if (!rest.trim().isEmpty()) {
                        for (String n : rest.split(",")) {
                            String trimmed = n.trim();
                            if (!trimmed.isEmpty() && trimmed.matches("\\d+")) {
                                neighbors.add(Integer.parseInt(trimmed));
                            }
                        }
                    }
                    
                    level0.put(nodeId, neighbors);
                    
                } catch (Exception e) {
                    continue; // Skip invalid lines
                }
            }
            
            data.connections.put(0, level0);
            
            // Validate vector count against actual parsed data
            if (data.vectorCount != level0.size()) {
                System.out.println("⚠️ Parsed vector count (" + data.vectorCount + ") differs from adjacency nodes (" + level0.size() + ")");
                System.out.println("📊 Using parsed adjacency count: " + level0.size());
                data.vectorCount = level0.size();
            }
            
            // Phase 3: Parse dataset vectors
            data.vectors = new float[data.vectorCount][];
            int vIdx = parseVectors(datasetVectorLines, data.vectors, "DATASET_VECTOR_", data);
            System.out.println("📊 Parsed " + vIdx + " dataset vectors");
            
            // Phase 4: Parse query vectors
            data.queryVectors = new float[data.queryCount][];
            int qIdx = parseVectors(queryVectorLines, data.queryVectors, "QUERY_VECTOR_", data);
            System.out.println("📊 Parsed " + qIdx + " query vectors");

            // Handle parsing failures with correct sizes
            if (vIdx == 0) {
                System.err.println("❌ No dataset vectors parsed! Creating dummy vectors...");
                for (int i = 0; i < data.vectorCount; i++) {
                    data.vectors[i] = generateTestVector(data.dimension, i);
                }
            }
            
            if (qIdx == 0) {
                System.err.println("❌ No query vectors parsed! Creating dummy queries...");
                for (int i = 0; i < data.queryCount; i++) {
                    data.queryVectors[i] = generateTestVector(data.dimension, i + 1000);
                }
            }
            
            data.size = data.vectorCount;
            data.numLevels = 1;
            data.entryNode = 0;
            data.maxConnections = 64;
        }
        
        return data;
    }

    /**
     * Convert parsed CAGRA data into Lucene's OnHeapHnswGraph format with validation
     */
    public static OnHeapHnswGraph reconstructGraph(GraphData data) {
        System.out.println("=== Graph Reconstruction with Validation ===");
        System.out.println("Vector count: " + data.vectorCount);
        System.out.println("Max connections: " + data.maxConnections);
        
        // Create single-level HNSW graph
        OnHeapHnswGraph graph = new OnHeapHnswGraph(data.maxConnections, data.vectorCount);
        
        // Add all nodes to level 0
        for (int nodeId = 0; nodeId < data.vectorCount; nodeId++) {
            try {
                graph.addNode(0, nodeId);
                if (nodeId == 0) {
                    graph.trySetNewEntryNode(0, 0);
                }
            } catch (Exception e) {
                System.err.println("Failed to add node " + nodeId);
            }
        }
        
        // Add connections with validation
        Map<Integer, List<Integer>> level0Connections = data.connections.get(0);
        if (level0Connections != null) {
            int validConnections = 0;
            int invalidConnections = 0;
            
            for (Map.Entry<Integer, List<Integer>> entry : level0Connections.entrySet()) {
                int nodeId = entry.getKey();
                List<Integer> neighbors = entry.getValue();
                
                // Skip nodes that don't exist in our vector space
                if (nodeId >= data.vectorCount) {
                    System.err.println("⚠️ Skipping node " + nodeId + " (>= " + data.vectorCount + ")");
                    continue;
                }
                
                if (neighbors.isEmpty()) continue;
                
                try {
                    NeighborArray neighborArray = graph.getNeighbors(0, nodeId);
                    
                    // Filter out invalid neighbor IDs
                    List<Integer> validNeighbors = new ArrayList<>();
                    for (Integer neighborId : neighbors) {
                        if (neighborId >= 0 && neighborId < data.vectorCount) {
                            validNeighbors.add(neighborId);
                            validConnections++;
                        } else {
                            System.err.println("⚠️ Skipping invalid neighbor " + neighborId + " for node " + nodeId);
                            invalidConnections++;
                        }
                    }
                    
                    addNeighborsToArray(neighborArray, validNeighbors);
                    
                } catch (Exception e) {
                    System.err.println("Failed to add neighbors for node " + nodeId + ": " + e.getMessage());
                }
            }
            
            System.out.println("✅ Valid connections: " + validConnections);
            System.out.println("❌ Invalid connections filtered: " + invalidConnections);
        }
        
        System.out.println("Graph reconstruction completed");
        return graph;
    }

    /**
     * Helper method to add neighbors to NeighborArray
     */
    private static void addNeighborsToArray(NeighborArray na, List<Integer> neighbors) {
        for (Integer neighbor : neighbors) {
            try {
                na.addOutOfOrder(neighbor.intValue(), 1.0f); // Use dummy score
            } catch (IllegalStateException e) {
                break; // NeighborArray is full
            } catch (Exception e) {
                break; // Other errors
            }
        }
    }

    /**
     * Create Lucene index with pre-built custom graph
     */
    private static void createIndexWithReconstructedData(GraphData data, OnHeapHnswGraph graph) throws IOException {
        System.out.println("=== Creating Index with Custom Graph ===");
        
        Path indexPath = Paths.get("complete-injected-index");
        if (java.nio.file.Files.exists(indexPath)) {
            deleteDirectory(indexPath.toFile());
        }
        
        Directory directory = FSDirectory.open(indexPath);
        IndexWriterConfig config = new IndexWriterConfig();
        
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            // Add first document to trigger FieldWriter creation
            Document firstDoc = new Document();
            firstDoc.add(new KnnFloatVectorField("vector", data.vectors[0], VectorSimilarityFunction.COSINE));
            firstDoc.add(new StringField("id", "injected_0", Field.Store.YES));
            writer.addDocument(firstDoc);
            
            // Inject custom graph
            Lucene99HnswVectorsWriter vectorWriter = Lucene99HnswVectorsWriter.getCurrentWriter();
            if (vectorWriter != null) {
                Lucene99HnswVectorsWriter.FieldWriter<?> fieldWriter = vectorWriter.getFieldWriter("vector");
                if (fieldWriter != null) {
                    fieldWriter.setCustomGraph(graph);
                    System.out.println("✅ Custom graph injected successfully!");
                }
            }
            
            // Add remaining documents
            for (int i = 1; i < data.vectorCount; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField("vector", data.vectors[i], VectorSimilarityFunction.COSINE));
                doc.add(new StringField("id", "injected_" + i, Field.Store.YES));
                writer.addDocument(doc);
            }
            
            writer.commit();
        }
        
        directory.close();
        System.out.println("✅ Index creation completed");
    }

    /**
     * Test the indexed data with actual CAGRA query vectors and compare results
     */
    public static void runQueryVectorTests(GraphData graphData) throws IOException {
        Path indexPath = Paths.get("complete-injected-index");
        if (!java.nio.file.Files.exists(indexPath)) {
            throw new IOException("Index not found: " + indexPath);
        }
        
        Directory directory = FSDirectory.open(indexPath);
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            System.out.println("✅ Index opened: " + reader.numDocs() + " docs");
            
            // Get actual vector dimension from index
            int actualDimension = -1;
            for (LeafReaderContext context : reader.leaves()) {
                LeafReader leafReader = context.reader();
                FloatVectorValues vectorValues = leafReader.getFloatVectorValues("vector");
                if (vectorValues != null) {
                    actualDimension = vectorValues.dimension();
                    System.out.println("✅ Vector field found, count=" + vectorValues.size() + ", dim=" + actualDimension);
                    break;
                }
            }
            
            if (actualDimension == -1) {
                System.err.println("❌ No vector field found!");
                return;
            }
            
            IndexSearcher searcher = new IndexSearcher(reader);
            
            // Test with all query vectors
            System.out.println("\n🔍 === Testing with " + graphData.queryCount + " CAGRA Query Vectors ===");
            
            for (int q = 0; q < graphData.queryCount && q < graphData.queryVectors.length; q++) {
                if (graphData.queryVectors[q] == null) continue;
                
                System.out.println("\n🔍 Query " + q + ":");
                System.out.println("  Query vector: " + Arrays.toString(Arrays.copyOf(graphData.queryVectors[q], Math.min(5, graphData.queryVectors[q].length))) + "...");
                
                TopDocs results = searcher.search(new KnnFloatVectorQuery("vector", graphData.queryVectors[q], 10), 10);
                System.out.println("  Found " + results.scoreDocs.length + " results:");
                
                for (int i = 0; i < Math.min(5, results.scoreDocs.length); i++) {
                    Document doc = searcher.storedFields().document(results.scoreDocs[i].doc);
                    float score = results.scoreDocs[i].score;
                    String docId = doc.get("id");
                    int originalId = Integer.parseInt(docId.replace("injected_", ""));
                    
                    System.out.printf("    %d. %s (original_id=%d, score=%.1f)\n", 
                        i + 1, docId, originalId, score);
                }
            }
            
            // Also run one test with a random vector for comparison
            System.out.println("\n🔍 === Comparison with Random Test Vector ===");
            float[] randomQuery = generateTestVector(actualDimension);
            TopDocs randomResults = searcher.search(new KnnFloatVectorQuery("vector", randomQuery, 5), 5);
            System.out.println("Random query found " + randomResults.scoreDocs.length + " results:");
            
            for (int i = 0; i < randomResults.scoreDocs.length; i++) {
                Document doc = searcher.storedFields().document(randomResults.scoreDocs[i].doc);
                float score = randomResults.scoreDocs[i].score;
                System.out.printf("  %d. %s (score %.5f)\n", i + 1, doc.get("id"), score);
            }
        }
        directory.close();
    }

    /**
     * Generate test vector for querying
     */
    private static float[] generateTestVector(int dimension) {
        return generateTestVector(dimension, 0);
    }

    private static float[] generateTestVector(int dimension, int seed) {
        Random random = new Random(42 + seed);
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = (float) random.nextGaussian();
        }
        
        // Normalize vector
        float norm = 0.0f;
        for (float v : vector) norm += v * v;
        norm = (float) Math.sqrt(norm);
        
        if (norm > 0) {
            for (int i = 0; i < dimension; i++) {
                vector[i] /= norm;
            }
        }
        return vector;
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

    /**
     * Utility to recursively delete a directory
     */
    private static void deleteDirectory(java.io.File dir) {
        if (dir.exists()) {
            java.io.File[] files = dir.listFiles();
            if (files != null) {
                for (java.io.File file : files) {
                    if (file.isDirectory()) {
                        deleteDirectory(file);
                    } else {
                        file.delete();
                    }
                }
            }
            dir.delete();
        }
    }

    /**
     * Helper method to parse vectors (works for both dataset and query vectors)
     */
    private static int parseVectors(List<String> vectorLines, float[][] targetArray, String vectorPrefix, GraphData data) {
        int vIdx = 0;
        
        for (int i = 0; i < vectorLines.size(); i++) {
            String vline = vectorLines.get(i);
            
            if (vline.startsWith(vectorPrefix)) {
                try {
                    String vectorId = vline.substring(0, vline.indexOf(":"));
                    
                    // Collect multi-line vector data
                    StringBuilder vecStrBuilder = new StringBuilder();
                    
                    int colon = vline.indexOf(":");
                    if (colon >= 0) {
                        String firstPart = vline.substring(colon + 1).trim();
                        if (!firstPart.isEmpty()) {
                            vecStrBuilder.append(firstPart);
                        }
                    }
                    
                    // Continue reading lines until next vector or end
                    int j = i + 1;
                    while (j < vectorLines.size()) {
                        String nextLine = vectorLines.get(j).trim();
                        if (nextLine.startsWith(vectorPrefix) || nextLine.startsWith("===")) {
                            break;
                        }
                        if (!nextLine.isEmpty()) {
                            if (vecStrBuilder.length() > 0) {
                                vecStrBuilder.append(" ");
                            }
                            vecStrBuilder.append(nextLine);
                        }
                        j++;
                    }
                    
                    String vecStr = vecStrBuilder.toString().trim();
                    if (vecStr.isEmpty()) continue;
                    
                    // Parse vector values
                    if (vecStr.startsWith("[")) vecStr = vecStr.substring(1);
                    if (vecStr.endsWith("]")) vecStr = vecStr.substring(0, vecStr.length() - 1);
                    
                    List<Float> validValues = new ArrayList<>();
                    for (String val : vecStr.split(",")) {
                        String trimmed = val.trim();
                        if (!trimmed.isEmpty()) {
                            try {
                                validValues.add(Float.parseFloat(trimmed));
                            } catch (NumberFormatException e) {
                                // Skip invalid values
                            }
                        }
                    }
                    
                    if (validValues.isEmpty()) continue;
                    
                    // Set dimension from first valid vector
                    if (data.dimension == -1) {
                        data.dimension = validValues.size();
                        System.out.println("📊 Detected vector dimension: " + data.dimension + " from " + vectorId);
                    }
                    
                    if (validValues.size() != data.dimension) continue;
                    
                    // Convert to float array
                    float[] vec = new float[validValues.size()];
                    for (int k = 0; k < validValues.size(); k++) {
                        vec[k] = validValues.get(k);
                    }
                    
                    if (vIdx < targetArray.length) {
                        targetArray[vIdx++] = vec;
                    }
                    
                    if (vIdx <= 3) {
                        System.out.println("📊 Parsed " + vectorId + " (" + vec.length + "D): " + 
                            Arrays.toString(Arrays.copyOf(vec, Math.min(5, vec.length))) + "...");
                    }
                    
                    i = j - 1; // Skip processed lines
                    
                } catch (Exception e) {
                    System.err.println("⚠️ Error parsing vector: " + vline);
                    continue;
                }
            }
        }
        
        if (data.vectors[0] != null) {
            System.out.println("📊 First vector sample: " + Arrays.toString(Arrays.copyOf(data.vectors[0], Math.min(10, data.vectors[0].length))));
            System.out.println("📊 First vector length: " + data.vectors[0].length);
        }
        
        return vIdx;
    }

    /**
     * Validate graph structure before using it
     */
    private static void validateGraphStructure(GraphData data) {
        System.out.println("🔍 === Graph Structure Validation ===");
        
        Map<Integer, List<Integer>> level0 = data.connections.get(0);
        if (level0 == null) {
            System.err.println("❌ No level 0 connections found!");
            return;
        }
        
        // Find min/max node IDs
        int minNodeId = Integer.MAX_VALUE;
        int maxNodeId = Integer.MIN_VALUE;
        int totalConnections = 0;
        
        for (Map.Entry<Integer, List<Integer>> entry : level0.entrySet()) {
            int nodeId = entry.getKey();
            List<Integer> neighbors = entry.getValue();
            
            minNodeId = Math.min(minNodeId, nodeId);
            maxNodeId = Math.max(maxNodeId, nodeId);
            totalConnections += neighbors.size();
            
            // Check if any neighbor IDs are out of bounds
            for (Integer neighborId : neighbors) {
                if (neighborId < 0 || neighborId >= data.vectorCount) {
                    System.err.println("❌ Node " + nodeId + " has invalid neighbor: " + neighborId + 
                        " (valid range: 0-" + (data.vectorCount - 1) + ")");
                }
                if (neighborId > maxNodeId) {
                    maxNodeId = neighborId;
                }
            }
        }
        
        System.out.println("📊 Parsed nodes: " + level0.size());
        System.out.println("📊 Node ID range: " + minNodeId + " to " + maxNodeId);
        System.out.println("📊 Expected vector count: " + data.vectorCount);
        System.out.println("📊 Total connections: " + totalConnections);
        
        // Check for gaps in node IDs
        Set<Integer> allReferencedNodes = new HashSet<>();
        for (Map.Entry<Integer, List<Integer>> entry : level0.entrySet()) {
            allReferencedNodes.add(entry.getKey());
            allReferencedNodes.addAll(entry.getValue());
        }
        
        System.out.println("📊 Unique nodes referenced: " + allReferencedNodes.size());
        if (!allReferencedNodes.isEmpty()) {
            System.out.println("📊 Max referenced node ID: " + Collections.max(allReferencedNodes));
        
            // Check if we have vectors for all referenced nodes
            int maxReferencedId = Collections.max(allReferencedNodes);
            if (maxReferencedId >= data.vectorCount) {
                System.err.println("❌ CRITICAL: Referenced node ID " + maxReferencedId + 
                    " exceeds vector count " + data.vectorCount);
                System.err.println("❌ This will cause ArrayIndexOutOfBoundsException!");
            }
        }
    }
}
