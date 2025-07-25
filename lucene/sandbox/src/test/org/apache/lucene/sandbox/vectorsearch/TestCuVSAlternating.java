/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.sandbox.vectorsearch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.QueryBuilder;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "prints info from within cuvs")
public class TestCuVSAlternating extends LuceneTestCase {

  static final Codec codec = TestUtil.alwaysKnnVectorsFormat(new CuVSVectorsFormat());
  static IndexSearcher searcher;
  static IndexReader reader;
  static Directory directory;
  static List<float[]> vectors;
  static List<Integer> docIdsWithVectors;

  static final int NUM_DOCS = 100;
  static final int DIMENSIONS = 128;
  static final int TOP_K = 10;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuvs not supported", CuVSVectorsFormat.supported());
    directory = newDirectory();

    RandomIndexWriter writer =
        new RandomIndexWriter(
            random(),
            directory,
            newIndexWriterConfig(new MockAnalyzer(random(), MockTokenizer.SIMPLE, true))
                .setMaxBufferedDocs(TestUtil.nextInt(random(), 100, 1000))
                .setCodec(codec)
                .setMergePolicy(newTieredMergePolicy()));

    Random random = random();
    vectors = new ArrayList<>();
    docIdsWithVectors = new ArrayList<>();

    for (int i = 0; i < NUM_DOCS; i++) {
      Document doc = new Document();
      doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
      doc.add(new TextField("quality", random.nextBoolean()? "good": "bad", Field.Store.YES));
      if (i % 2 == 0) {
        float[] vector = new float[DIMENSIONS];
        for (int j = 0; j < DIMENSIONS; j++) {
          vector[j] = random.nextFloat() * 100;
        }
        doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.EUCLIDEAN));
        vectors.add(vector);
        docIdsWithVectors.add(i);
      }
      
      writer.addDocument(doc);
    }

    reader = writer.getReader();
    searcher = newSearcher(reader);
    writer.close();
  }

  @AfterClass
  public static void afterClass() throws Exception {
    if (reader != null) reader.close();
    if (directory != null) directory.close();
    searcher = null;
    reader = null;
    directory = null;
  }

  @Test
  public void testAlternatingVectorSearch() throws IOException {
    Random random = random();
    float[] queryVector = new float[DIMENSIONS];
    for (int i = 0; i < DIMENSIONS; i++) {
      queryVector[i] = random.nextFloat() * 100;
    }

    Query regularQuery = new TermQuery(new Term("quality", "bad"));
    Query vectorQuery = new KnnFloatVectorQuery("vector", queryVector, 10);
    Query booleanQuery = new BooleanQuery.Builder().add(regularQuery, Occur.FILTER).add(vectorQuery, Occur.MUST).build();
    
    ScoreDoc[] hits = searcher.search(booleanQuery,100).scoreDocs;

    for (int i=0; i<hits.length; i++) {
      System.out.println("Doc: " + hits[i].doc);
    }
    assertEquals("Should return exactly " + TOP_K + " results", TOP_K, hits.length);

    List<Integer> expectedTopK = computeExpectedTopK(queryVector);

    for (ScoreDoc hit : hits) {
      int docId = Integer.parseInt(reader.storedFields().document(hit.doc).get("id"));
      assertTrue("Document " + docId + " should have a vector (even numbered)", docId % 2 == 0);
      assertTrue("Document " + docId + " should be in expected top-k results", expectedTopK.contains(docId));
    }
    System.out.println("********************************************");
  }

  private List<Integer> computeExpectedTopK(float[] queryVector) {
    Map<Integer, Double> distances = new TreeMap<>();
    
    for (int i = 0; i < vectors.size(); i++) {
      double distance = 0;
      float[] vector = vectors.get(i);
      for (int j = 0; j < DIMENSIONS; j++) {
        distance += (queryVector[j] - vector[j]) * (queryVector[j] - vector[j]);
      }
      distances.put(docIdsWithVectors.get(i), distance);
    }

    return distances.entrySet().stream()
        .sorted(Map.Entry.comparingByValue())
        .limit(TOP_K)
        .map(Map.Entry::getKey)
        .toList();
  }
}